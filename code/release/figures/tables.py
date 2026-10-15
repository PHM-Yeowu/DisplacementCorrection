"""Table 4: the errors of every run of grid.py for each seed, with their mean and standard deviation.

    python tables.py      writes paper/tables/seeds.tex from results/errors.json (results.py)

Double rules at the top and the bottom (the macros dtoprule and dbottomrule of main.tex), a single rule below the
header and between the benchmarks, no vertical rules, and the proposed correction in the last rows.
"""

import os

import numpy as np

from figlib import NAME, V2
from results import EXAMPLES, SEEDS, errors, mean_std

OUT = os.path.join(V2, "paper", "tables")
LABEL = (("backbone", "Backbone"), ("ctl", "Continued training"),
         ("addfull1", "MSNN, one stage"), ("addfull2", "MSNN, two stages"),
         ("dispfull1", "Displacement, one stage"), ("dispfull2", "Displacement, two stages"))


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def cell(v):
    m, s = mean_std(v)
    e = int(np.floor(np.log10(m)))
    return rf"$({m / 10**e:.2f} \pm {s / 10**e:.2f})\times10^{{{e}}}$"


def main():
    err = {(r["example"], r["method"], r["seed"]): r["error"] for r in errors()}
    lines = ["\\begin{tabular}{l" + "c" * (len(SEEDS) + 1) + "}", "\\dtoprule",
             "Method & " + " & ".join(f"Seed {s}" for s in SEEDS)
             + " & \\makecell{Mean $\\pm$\\\\ standard deviation} \\\\", "\\midrule"]
    for i, ex in enumerate(EXAMPLES):
        if i:
            lines.append("\\midrule")
        lines.append(rf"\multicolumn{{{len(SEEDS) + 2}}}{{l}}{{\textit{{{NAME[ex]}}}}} \\")
        for k, label in LABEL:
            v = [err[(ex, k, s)] for s in SEEDS]
            lines.append(rf"\quad {label} & " + " & ".join(map(sci, v)) + f" & {cell(v)} \\\\")
    lines += ["\\dbottomrule", "\\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seeds.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUT, "seeds.tex"))


if __name__ == "__main__":
    main()
