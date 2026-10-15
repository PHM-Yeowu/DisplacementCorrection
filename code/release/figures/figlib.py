"""Style and paths shared by results.py, fig_curves.py and tables.py.

Style follows the figure rules of the paper: Times New Roman text with STIX math, axes of ratio 4:3, ticks at
both ends of every axis, no unit labels (every quantity is dimensionless), and one colour, line style and marker
per method so that the curves stay distinct in grey-scale print. Figures are drawn at their final width (6.5 in
for the text width), so the figure text is 9-10 pt against the 10 pt body text.
"""

import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS = os.path.join(V2, "runs")
Q = os.path.join(RUNS, "queue")
RESULTS = os.path.join(V2, "results")
OUT = os.path.join(V2, "paper", "figures")
FULL = 6.5  # \textwidth in inches

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.grid": False, "axes.unicode_minus": True,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# the proposed correction in black, MSNN in blue, continued training in orange and the backbone a dotted line
COLOR = {"disp": "#000000", "add": "#0072B2", "ctl": "#D55E00", "bb": "0.45"}
STYLE = {"disp": ("-", "o"), "add": ("--", "s"), "ctl": ("-.", "^"), "bb": (":", None)}
LABEL = {"disp": "Displacement", "add": "MSNN", "ctl": "Continued training", "bb": "Backbone"}
TITLE = {"kdv": "Korteweg–de Vries", "burgers": r"Burgers, $\nu = 0.01/\pi$",
         "burgers_nu3": r"Burgers, $\nu = 0.003/\pi$"}
NAME = {"kdv": "KdV", "burgers": r"Burgers, $\nu = 0.01/\pi$", "burgers_nu3": r"Burgers, $\nu = 0.003/\pi$"}


def finished(job):
    return os.path.exists(os.path.join(Q, job + ".done"))


def log_errors(job, keys):
    """{step: error} from runs/queue/<job>.log for the first of `keys` (error/<key>) that the log carries."""
    logged = {k: {} for k in keys}
    step = None
    for line in open(os.path.join(Q, f"{job}.log")):
        m = re.search(r"Iter:\s+(\d+)", line)
        if m:
            step = int(m.group(1))
        m = re.search(r"error/(\w+)\s+([0-9.eE+-]+)", line)
        if m and m.group(1) in logged and step is not None:
            logged[m.group(1)][step] = float(m.group(2))
    return next((logged[k] for k in keys if logged[k]), {})


def panel(k, title=""):
    return f"({'abcdefghijkl'[k]}) {title}".rstrip()


def decades(lo, hi):
    """Logarithmic limits at whole decades around [lo, hi] and a tick at every decade."""
    a, b = np.floor(np.log10(lo)), np.ceil(np.log10(hi))
    return 10.0 ** a, 10.0 ** b, 10.0 ** np.arange(a, b + 0.5)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".pdf"))
