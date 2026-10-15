"""The errors of every finished run of grid.py, collected into results/ so that fig_curves.py and tables.py need
no runs.

    python results.py     writes results/errors.json and results/curves.json and prints the mean errors

errors.json has one entry per run with the final error. Its method is backbone, ctl (continued training),
addfull1 / addfull2 (one / two stages of MSNN) or dispfull1 / dispfull2 (of the displacement correction).
curves.json has the error against the iterations of both stages of each correction and of continued training, with
the evaluation of the final checkpoint as the last point of continued training. Errors are relative L2 errors
against the converged reference where there is one (burgers, make_references.py), else against the reference data
of the example.
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from figlib import RESULTS, RUNS, finished, log_errors

EXAMPLES = ("kdv", "burgers", "burgers_nu3")
SEEDS = (42, 43, 44)
STAGE = 40000  # iterations of one stage
METHODS = ("backbone", "ctl", "addfull1", "addfull2", "dispfull1", "dispfull2")
NAME = re.compile(r"^(c1|c2)_(\w+?)_s(\d+)_(add|disp)_?full$")


def error(d):
    return d["error/true"] if "error/true" in d else d["error/ref"]


def collect_errors():
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "corrections", "c[12]_*", "result.json"))):
        name = os.path.basename(os.path.dirname(path))
        m = NAME.match(name)
        if not m or m.group(2) not in EXAMPLES or not finished(name):
            continue
        stage, example, seed, form = m.groups()
        r = json.load(open(path))
        out.append(dict(example=example, seed=int(seed), method=f"{form}full{stage[1]}", error=error(r["after"])))
        if stage == "c1" and form == "disp":  # the error before the first stage is that of the backbone
            out.append(dict(example=example, seed=int(seed), method="backbone", error=error(r["before"])))
    for example in EXAMPLES:
        for seed in SEEDS:
            path = os.path.join(RUNS, "controls", f"{example}_s{seed}_plus40k", "eval.json")
            if finished(f"ev_ctl_{example}_s{seed}"):
                out.append(dict(example=example, seed=seed, method="ctl", error=error(json.load(open(path)))))
    return out


def collect_curves():
    out = {}
    for example in EXAMPLES:
        for seed in SEEDS:
            for form in ("add", "disp"):
                names = (f"c1_{example}_s{seed}_{form}_full", f"c2_{example}_s{seed}_{form}full")
                if all(finished(n) for n in names):
                    s1, s2 = (log_errors(n, ("true", "ref")) for n in names)
                    steps = sorted(s1) + [STAGE + k for k in sorted(s2)[1:]]
                    errors = [s1[k] for k in sorted(s1)] + [s2[k] for k in sorted(s2)[1:]]
                    out[f"{example}_s{seed}_{form}"] = dict(iteration=steps, error=errors)
            if finished(f"ev_ctl_{example}_s{seed}"):
                d = log_errors(f"ctl_{example}_s{seed}_plus40k", ("true", "l2"))
                ev = json.load(open(os.path.join(RUNS, "controls", f"{example}_s{seed}_plus40k", "eval.json")))
                steps = sorted(d)
                out[f"{example}_s{seed}_ctl"] = dict(iteration=[k - steps[0] for k in steps] + [STAGE],
                                                     error=[d[k] for k in steps] + [error(ev)])
    return out


def errors():
    with open(os.path.join(RESULTS, "errors.json")) as f:
        return json.load(f)


def curves():
    with open(os.path.join(RESULTS, "curves.json")) as f:
        return json.load(f)


def mean_std(v):
    """Mean and sample standard deviation (ddof 1) over the seeds."""
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = collect_errors()
    with open(os.path.join(RESULTS, "errors.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(RESULTS, "curves.json"), "w") as f:
        json.dump(collect_curves(), f)
    t = defaultdict(list)
    for r in rows:
        t[(r["example"], r["method"])].append(r["error"])
    print("| example | " + " | ".join(METHODS) + " |")
    print("|---" * (len(METHODS) + 1) + "|")
    for ex in EXAMPLES:
        cells = ["{:.2e} ± {:.1e}".format(*mean_std(t[(ex, m)])) if t[(ex, m)] else "" for m in METHODS]
        print(f"| {ex} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
