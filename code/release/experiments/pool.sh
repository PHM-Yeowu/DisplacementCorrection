#!/bin/bash
# Persistent job pool over GPUs and CPU slots, with priorities and dependencies.
#
#   pool.sh worker <slot> [need_mib] [pattern]   start one worker (runs until runs/queue/STOP2 exists);
#                                        with a pattern (grep -E on job names) it takes only matching jobs
#   pool.sh add <prio> <name> <command>  add a job; lower prio runs first; only jobs with prio up to
#                                        the number in runs/queue/max_prio (if present) are started
#   pool.sh after <deps> <prio> <name> <command>   same, but only once every job in the
#                                        comma-separated <deps> has finished successfully
#
# A slot is a GPU index or cpu:<first>-<last> (a core range for one CPU job). A GPU worker waits
# until its GPU has the free memory in runs/queue/min_free_mib (else need_mib), the host has the available
# memory in runs/queue/min_host_mib (else 40000 MiB), and fewer of our own
# compute processes than the cap, both read each time, the cap from runs/queue/max_per_gpu (else
# MAX_PER_GPU, default 2), and holds the GPU for 90 s
# after starting a job, so that two workers never start on the same free memory. Jobs live in runs/queue/pending/<prio>_<name>.job,
# move to running/ when taken and to finished/ at the end; logs are runs/queue/<name>.log and a
# successful job leaves runs/queue/<name>.done. A crashed job can still exit 0: check its log.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)            # the repository root
Q=$ROOT/runs/queue; mkdir -p "$Q/pending" "$Q/running" "$Q/finished" "$Q/holds"
LOCK=$Q/pool.lock; touch "$LOCK"

add_job() {  # deps prio name command...
  local deps=$1 prio=$2 name=$3; shift 3
  printf '# after: %s\n%s\n' "$deps" "$*" > "$Q/pending/.$name.tmp"
  mv "$Q/pending/.$name.tmp" "$Q/pending/$(printf '%03d' "$prio")_$name.job"
}

take_job() {  # prints the name of a job whose dependencies are done, and moves it to running/
  local f name deps d ok
  for f in $(ls "$Q/pending" 2>/dev/null | grep '\.job$' | sort); do
    name=${f#*_}; name=${name%.job}
    echo "$name" | grep -qE "$PATTERN" || continue
    [ "$((10#${f%%_*}))" -le "$(cat "$Q/max_prio" 2>/dev/null || echo 999)" ] || continue
    deps=$(sed -n 's/^# after: *//p' "$Q/pending/$f") || continue  # unreadable for a moment on the NAS: skip
    deps=$(echo "$deps" | tr ',' ' ')
    ok=1
    for d in $deps; do [ "$d" = "-" ] && continue; [ -f "$Q/$d.done" ] || { ok=0; break; }; done
    if [ $ok = 1 ] && mv "$Q/pending/$f" "$Q/running/$name.job"; then
      # a file can stay unreadable under its new name on the NAS: back to pending/, where the watcher renews it
      sed -n '2p' "$Q/running/$name.job" 2>/dev/null | grep -q . || { mv "$Q/running/$name.job" "$Q/pending/$f"; continue; }
      printf '# prio: %s_\n' "${f%%_*}" >> "$Q/running/$name.job"  # kept for a requeue
      echo "$name"; return
    fi
  done
}

worker() {
  local slot=$1 need=${2:-6000} name cmd pre free
  PATTERN=${3:-.}
  while [ ! -f "$Q/STOP2" ]; do
    if [[ $slot != cpu:* ]]; then  # wait for memory and for fewer than the cap of jobs on the GPU
      need=$(cat "$Q/min_free_mib" 2>/dev/null || echo "$need")
      free=$(nvidia-smi --id=$slot --query-gpu=memory.free --format=csv,noheader,nounits)
      procs=$(for p in $(nvidia-smi --id=$slot --query-compute-apps=pid --format=csv,noheader); do
                ps -o user= -p "$p"; done | grep -c "^$(whoami)$")  # our own jobs only
      cap=$(cat "$Q/max_per_gpu" 2>/dev/null || echo "${MAX_PER_GPU:-2}")
      # host memory too: when other jobs fill the RAM, every process on the machine stalls in reclaim
      host=$(awk '/^MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
      host_need=$(cat "$Q/min_host_mib" 2>/dev/null || echo 40000)
      if [ "$free" -lt "$need" ] || [ "$host" -lt "$host_need" ] || [ "$procs" -ge "$cap" ] \
         || ! mkdir "$Q/holds/gpu$slot" 2>/dev/null; then
        sleep 30; continue
      fi
    fi
    name=$(flock "$LOCK" bash -c "$(declare -f take_job); Q='$Q'; PATTERN='$PATTERN'; take_job")
    if [ -z "$name" ]; then
      [[ $slot != cpu:* ]] && rmdir "$Q/holds/gpu$slot"
      sleep 30; continue
    fi
    if [[ $slot == cpu:* ]]; then
      pre="env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu taskset -c ${slot#cpu:}"
    else
      ( sleep 90; rmdir "$Q/holds/gpu$slot" ) &
      pre="env CUDA_VISIBLE_DEVICES=$slot XLA_PYTHON_CLIENT_PREALLOCATE=false"
    fi
    cmd=$(sed -n '2,$p' "$Q/running/$name.job")
    if [ -z "$cmd" ]; then  # never run an empty command: it would exit 0 and leave a .done
      echo "[$(date '+%m-%d %H:%M')] unreadable, requeued $name"
      mv "$Q/running/$name.job" "$Q/pending/050_${name}.job"
      sleep 30; continue
    fi
    echo "[$(date '+%m-%d %H:%M')] start $name on $slot"
    # with POOL_SLICE set, every job runs in that systemd user slice and may use at most JOB_MEM_MAX and no swap
    if [ -n "${POOL_SLICE:-}" ]; then
      systemd-run --user --scope --quiet --slice="$POOL_SLICE" -p MemorySwapMax=0 -p MemoryMax="${JOB_MEM_MAX:-16G}" \
        $pre bash -c "$cmd" > "$Q/$name.log" 2>&1
    else
      $pre bash -c "$cmd" > "$Q/$name.log" 2>&1
    fi
    rc=$?
    [ $rc -eq 0 ] && touch "$Q/$name.done"
    if [ ! -f "$Q/$name.done" ] && { [ $rc -eq 137 ] || grep -qE "RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|Failed to get configs|Autotuning failed|Unable to initialize backend .cuda.|gpusolverDnCreate|cuSolver internal error" "$Q/$name.log"; }; then
      # out of GPU memory (the GPUs are shared; XLA's autotuner also fails without memory, and so does the
      # start of CUDA on a full GPU), or ended by the memory cap of the slice (exit 137): back to the queue,
      # retried later
      echo "[$(date '+%m-%d %H:%M')] out of memory, requeued $name"
      mv "$Q/$name.log" "$Q/$name.oom.log"
      prio=$(sed -n 's/^# prio: *//p' "$Q/running/$name.job" | head -1)
      mv "$Q/running/$name.job" "$Q/pending/${prio:-050_}${name}.job"
      sleep 300; continue
    fi
    echo "[$(date '+%m-%d %H:%M')] end $name on $slot"
    mv "$Q/running/$name.job" "$Q/finished/$name.job"
  done
}

case ${1:-} in
  worker) worker "$2" "${3:-6000}" "${4:-.}" ;;
  add) add_job - "$2" "$3" "${@:4}" ;;
  after) add_job "$2" "$3" "$4" "${@:5}" ;;
  *) echo "usage: pool.sh worker <slot> [need_mib] | add <prio> <name> <cmd> | after <deps> <prio> <name> <cmd>"; exit 1 ;;
esac
