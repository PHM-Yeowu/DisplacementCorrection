"""Every run of the paper as jobs for pool.sh.

    python grid.py            prints the jobs not yet done, queued or running
    python grid.py --submit   adds them

For kdv, burgers and burgers_nu3 and the seeds 42, 43, 44: the PirateNet backbone of the jaxpi2 recipe, two stages
of the displacement correction and of MSNN on it, and the backbone trained for 40k more steps (continued training)
with the errors of the result.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = os.path.join(V2, "envs", "v2", "bin", "python")
POOL = os.path.join(HERE, "pool.sh")
Q = os.path.join(V2, "runs", "queue")
BB, CORR = os.path.join(V2, "runs", "backbones"), os.path.join(V2, "runs", "corrections")
REFS = os.path.join(V2, "runs", "references")

EXAMPLES = {"kdv": "pseudo_time", "burgers": "baseline", "burgers_nu3": "baseline"}  # example: jaxpi2 config
SEEDS = (42, 43, 44)
FORMS = ("disp", "add")
BB_STEPS = {"kdv": 50000}  # steps of the recipe, 1e5 for burgers
STEPS = 40000


def known(name):
    if os.path.exists(os.path.join(Q, name + ".done")) or os.path.exists(os.path.join(Q, "running", name + ".job")):
        return True
    return any(f.split("_", 1)[1] == name + ".job" for f in os.listdir(os.path.join(Q, "pending")))


def true_error(example):
    # burgers: its released reference is not converged, so the log also carries the error against make_references.py
    path = os.path.join(REFS, f"{example}_true.npz")
    return f"ERROR_TRUE={path} " if os.path.exists(path) else ""


def backbone(example, config, run, seed, extra=""):
    return (f"{true_error(example)}WANDB_MODE=offline {PY} {HERE}/run_jaxpi2.py {example} {config} {run} "
            f"--config.seed={seed} --config.training.resume=True {extra}")


def correction(example, bb, out, form, prev=None):
    p = f" --prev {prev}" if prev else ""
    return (f"WANDB_MODE=offline {PY} {HERE}/run_correction.py --example {example} --backbone {bb} "
            f"--out {out} --form {form} --steps {STEPS} --log_every 500{p}")


def jobs():
    out = []  # (prio, deps, name, cmd)
    for ex, cfg in EXAMPLES.items():
        for s in SEEDS:
            name = f"{ex}_s{s}"
            bb, bbdir = f"bb_{name}", f"{BB}/{name}"
            out.append((5, "-", bb, backbone(ex, cfg, bbdir, s)))
            for form in FORMS:
                c1, c2 = f"c1_{name}_{form}_full", f"c2_{name}_{form}full"
                out.append((10, bb, c1, correction(ex, bbdir, f"{CORR}/{c1}", form)))
                out.append((16, c1, c2, correction(ex, bbdir, f"{CORR}/{c2}", form, prev=f"{CORR}/{c1}")))
            # continued training, in a copy of the backbone's checkpoints
            ctl, ctldir = f"ctl_{name}_plus40k", f"{V2}/runs/controls/{name}_plus40k"
            steps = BB_STEPS.get(ex, 100000) + STEPS
            out.append((14, bb, ctl, f"mkdir -p {ctldir} && cp -rn {bbdir}/ckpt {ctldir}/ && "
                        + backbone(ex, cfg, ctldir, s, f"--config.training.max_steps={steps}")))
            out.append((4, ctl, f"ev_ctl_{name}", f"{PY} {HERE}/eval_backbone.py {ex} {ctldir}"))
    return out


def main():
    submit = "--submit" in sys.argv
    n = 0
    for prio, deps, name, cmd in jobs():
        if known(name):
            continue
        n += 1
        if submit:
            subprocess.run([POOL, "after", deps, str(prio), name, cmd], check=True)
        else:
            print(prio, deps, name)
    print(f"{n} jobs {'submitted' if submit else 'to submit'}")


if __name__ == "__main__":
    main()
