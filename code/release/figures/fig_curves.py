"""Figure 3: relative L2 error against the iterations of two correction stages, one panel per benchmark.

    python fig_curves.py

From results/curves.json (results.py). MSNN and the displacement correction over both stages on a linear iteration
axis, the second stage from 4e4 iterations, and continued training of the backbone for as many iterations as one
stage. Every curve is the mean over the seeds; the backbone is a dotted line. The fourth cell holds the legend.
"""

import numpy as np

from figlib import COLOR, FULL, LABEL, STYLE, TITLE, decades, panel, plt, save
from results import EXAMPLES, SEEDS, STAGE, curves


def mean_curve(cs):
    """Mean over the seeds on the iterations logged by every seed."""
    cs = [(np.array(c["iteration"]), np.array(c["error"])) for c in cs]
    grid = np.unique(np.concatenate([c[0] for c in cs]))
    grid = grid[grid <= min(c[0][-1] for c in cs)]
    return grid, np.mean([np.interp(grid, c[0], c[1]) for c in cs], 0)


def main():
    data = curves()
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.9), constrained_layout=True)
    for k, (ax, ex) in enumerate(zip(axes.ravel(), EXAMPLES)):
        values = []
        # markers by distance along the corrections and every 5000 iterations along continued training
        for key, every in (("ctl", (0, 50)), ("add", (0.02, 0.1)), ("disp", (0.07, 0.1))):
            c = mean_curve([data[f"{ex}_s{s}_{key}"] for s in SEEDS])
            values += [c[1].min(), c[1].max()]
            ls, marker = STYLE[key]
            ax.semilogy(c[0], c[1], color=COLOR[key], ls=ls, lw=1.1 if key in ("add", "disp") else 0.9,
                        marker=marker, ms=3.5, markevery=every, markerfacecolor="white", label=LABEL[key])
        bb = np.mean([data[f"{ex}_s{s}_disp"]["error"][0] for s in SEEDS])
        ax.axhline(bb, color=COLOR["bb"], ls=STYLE["bb"][0], lw=1.0, label=LABEL["bb"], zorder=1)
        values.append(bb)
        lo, hi, yt = decades(min(values), max(values) * 1.5)
        ax.set_ylim(lo, hi)
        ax.set_yticks(yt)
        ax.minorticks_off()
        ax.axvline(STAGE, color="0.5", ls="--", lw=0.7, zorder=0)
        for x0, text in ((STAGE / 2, "Stage 1"), (1.5 * STAGE, "Stage 2")):
            ax.text(x0, 0.96, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.set_xlim(0, 2 * STAGE)
        ax.set_xticks(np.arange(0, 2 * STAGE + 1, 2e4))
        ax.set_xticklabels(["0", "2", "4", "6", "8"])
        ax.set_xlabel(r"Iterations ($\times 10^4$)")
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.set_title(panel(k, TITLE[ex]))
        ax.set_box_aspect(0.75)
    leg = axes[1, 1]
    leg.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(LABEL[k]) for k in ("bb", "ctl", "add", "disp")]
    leg.legend([handles[i] for i in order], [labels[i] for i in order], loc="center", frameon=False)
    save(fig, "curves")


if __name__ == "__main__":
    main()
