"""Rebuild Figure 7 (result_actual_training_time) from the recorded evidence tables.

Figure-only script: reads quality_time_by_seed.csv and quality_time_summary.csv and
writes result_actual_training_time.{pdf,svg,png}. It writes no table or evidence file.
Copied from plot_time() in revision2_analysis/build_existing_evidence.py; only font sizes changed.

Run:  python manuscript_source/scripts/build_time_figure.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "revision2_preparation"
FIGURES = ROOT / "manuscript_source" / "figures"


def plot_time(figures, rows, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib._mathtext as _mt
    _mt.SHRINK_FACTOR = 0.95   # sub/superscripts at 0.95 x base: 7.6 pt for 8 pt text
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Single-column layout (Elsevier two-column page): 3.3 in wide, printed at 100% scale.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .5,
                         "lines.linewidth": 1.0, "savefig.facecolor": "white",
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": .5,
                         "ytick.major.width": .5, "xtick.major.pad": 2, "ytick.major.pad": 2,
                         "axes.labelpad": 2, "axes.titlepad": 3, "legend.handletextpad": .4,
                         "legend.columnspacing": 1.0, "legend.borderaxespad": .3})
    colors = {"Independent": "#0072B2", "MTT": "#009E73"}
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 5.3), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=.02, h_pad=.02, hspace=.04, wspace=.04)
    for ax, e in zip(axes, (2, 4)):
        for strategy in ("Independent", "MTT"):
            color = colors[strategy]
            for policy in ("raw", "matched_BN"):
                marker = "o" if policy == "raw" else "s"
                face = "white" if policy == "raw" else color
                individual = [r for r in rows if r["strategy"] == strategy and r["inference_T"] == e and r["state_policy"] == policy]
                ax.scatter([r["training_seconds"] for r in individual], [r["ce"] for r in individual], s=36,
                           marker=marker, facecolors=face, edgecolors=color, linewidths=.5, alpha=.25, zorder=1)
                data = sorted([r for r in summaries if r["strategy"] == strategy and r["inference_T"] == e and r["state_policy"] == policy],
                              key=lambda r: r["updates_per_trajectory"])
                means = np.array([r["ce_mean"] for r in data])
                errors = np.array([[r["ce_mean"] - r["ce_pointwise_95_low"] for r in data],
                                   [r["ce_pointwise_95_high"] - r["ce_mean"] for r in data]])
                ax.errorbar([r["mean_training_seconds"] for r in data], means, yerr=errors,
                            fmt=marker, linestyle="none", color=color, markerfacecolor=face,
                            markersize=6.0, markeredgewidth=1, capsize=2.5, elinewidth=.8, zorder=3)
                # Budget annotations identify actual saved states, not a connecting performance curve.
                if policy == "raw":
                    for row in data:
                        # MTT: label left of the raw marker. Independent: label centred below the
                        # matched-BN square of the same budget (the lowest marker at that x).
                        if strategy == "MTT":
                            xy = (row["mean_training_seconds"], row["ce_mean"])
                            kw = dict(xytext=(-5, 0), ha="right", va="center")
                        else:
                            bn = [r for r in summaries if r["strategy"] == strategy and r["inference_T"] == e
                                  and r["state_policy"] == "matched_BN" and r["updates_per_trajectory"] == row["updates_per_trajectory"]][0]
                            xy = (row["mean_training_seconds"], bn["ce_pointwise_95_low"])
                            kw = dict(xytext=(0, -4), ha="center", va="top")
                        ax.annotate(f"{row['updates_per_trajectory']:,}", xy, textcoords="offset points",
                                    fontsize=7, color=color, **kw)
        ax.set_title(f"({'a' if e == 2 else 'b'}) Inference window $e={e}$", loc="left")
        if ax is axes[-1]:
            ax.set_xlabel("Cumulative selected-strategy\ntraining time (s)")
        ax.set_ylabel("Test cross-entropy")
        ax.grid(axis="y", color="#DDDDDD", lw=.5)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1470)
        ax.margins(y=.2)
    handles = [Line2D([], [], linestyle="none", marker="o" if policy == "raw" else "s",
                      markerfacecolor="white" if policy == "raw" else colors[strategy], markeredgecolor=colors[strategy],
                      markersize=6.0, label=strategy + (", raw" if policy == "raw" else ", matched BN"))
               for strategy in ("Independent", "MTT") for policy in ("raw", "matched_BN")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, columnspacing=1.2, handletextpad=.4)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(figures / f"result_actual_training_time.{suffix}", dpi=240)
    plt.close(fig)


def main():
    rows = pd.read_csv(EVIDENCE / "quality_time_by_seed.csv").to_dict("records")
    summaries = pd.read_csv(EVIDENCE / "quality_time_summary.csv").to_dict("records")
    plot_time(FIGURES, rows, summaries)


if __name__ == "__main__":
    main()
