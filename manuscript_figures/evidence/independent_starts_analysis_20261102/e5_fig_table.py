"""Fig. 7 and Table 9 from the E5 per-start tables (K = 2048). Reads only the CSVs written by e5_starts.py."""
import csv, os, sys, math, json
import numpy as np
import matplotlib
import matplotlib._mathtext as _mt; _mt.SHRINK_FACTOR = 0.95; _mt.FontConstantsBase.sub1 = 0.42; _mt.FontConstantsBase.sub2 = 0.6; _mt.DejaVuSansFontConstants.sub1 = 0.42; _mt.DejaVuSansFontConstants.sub2 = 0.6
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding="utf-8")
# paths relative to this file: <root>/evidence/independent_starts_analysis_20261102/e5_fig_table.py
OUT = os.path.dirname(os.path.abspath(__file__))
EV = os.path.dirname(OUT)
FIGDIR = os.path.join(os.path.dirname(EV), "manuscript_source", "figures")
TABDIR = os.path.join(os.path.dirname(EV), "manuscript_source", "tables")

def f(x):
    try: return float(x)
    except Exception: return float("nan")

T9 = list(csv.DictReader(open(os.path.join(OUT, "table9_K2048_per_start.csv"), encoding="utf-8")))
for r in T9:
    for k in r:
        if k not in ("start", "cls", "dataset", "arch"):
            r[k] = f(r[k])

CLS = ["CIFAR-100, 30-epoch", "CIFAR-100, 600-epoch", "saturated, 600-epoch", "PlantWild, 600-epoch"]
CLS_LABEL = {"CIFAR-100, 30-epoch": "CIFAR-100, 30-epoch start (6)", "CIFAR-100, 600-epoch": "CIFAR-100, 600-epoch start (4)",
             "saturated, 600-epoch": "CIFAR-10, BloodMNIST64, EuroSAT (9)", "PlantWild, 600-epoch": "PlantWild (3)"}
COL = {"CIFAR-100, 30-epoch": "#1f77b4", "CIFAR-100, 600-epoch": "#d62728", "saturated, 600-epoch": "#7f7f7f", "PlantWild, 600-epoch": "#2ca02c"}
MK = {"SpikingResformer": "o", "Spikformer": "s"}

# Single-column layout (Elsevier two-column page): 3.3 in (238 pt) wide, printed at 100% scale.
FIGW = 3.3
# Every in-figure text is >= 7 pt: ticks and legend 7 pt, axis labels and panel titles 8 pt
# (math sub/superscripts render at 0.95 x 8 pt = 7.6 pt). Every marker is 6.0 pt wide (>= 5.6 pt).
FS_TICK, FS_LABEL, FS_LEG = 7.0, 8.0, 7.0
MK_MS = 6.0; MK_S = MK_MS ** 2
plt.rcParams.update({"font.size": FS_LABEL, 'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_LABEL,
                     "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, 'legend.fontsize': FS_LEG,
                     "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": .5, "ytick.major.width": .5,
                     "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 3,
                     "legend.handletextpad": .4, "legend.columnspacing": 1.0, "legend.borderaxespad": .3})
# Rows: (a) | (b) side by side; (c) full width; (d) full width sharing (c)'s categories and y range.
fig = plt.figure(figsize=(FIGW, 7.4), layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.04, hspace=.04)
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.35])
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :]); ax_d = fig.add_subplot(gs[2, :], sharex=ax_c, sharey=ax_c)

def scatter(ax, xkey, ykey, xlabel, ylabel, title):
    for r in T9:
        ax.scatter(r[xkey], r[ykey], s=MK_S, marker=MK[r["arch"]], facecolor=COL[r["cls"]], edgecolor="black", linewidth=0.4, zorder=3)
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=FS_LABEL, loc="left")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
    ax.margins(x=.08)

# Line breaks only (text unchanged): constrained layout ignores the horizontal overhang of
# titles and x labels, so each line must fit the ~80 pt-wide panel.
scatter(ax_a, "S0_acc_T2", "P2_acc_raw", "Retained start\n$S_0$: test\naccuracy at\n$e=2$ (%)",
        "Raw $P_2$ accuracy\ngain (pp)", "(a) Two-step source\nvs. reused\nfour-step source")
scatter(ax_b, "S0_acc_T4", "G44_acc_raw", "Retained start\n$S_0$: test\naccuracy at\n$e=4$ (%)",
        "Raw $G_{4,4}$ accuracy\ngain (pp)", "(b) Four-step\ncontinuation vs.\nretained start")

# (c) P2 CE: raw BN, raw vs calibrated; (d) matched BN, raw vs calibrated
order = list(range(len(T9)))
xs = np.arange(len(T9))
def ce_panel(ax, k_raw, k_cal, title):
    for i, r in enumerate(T9):
        c = COL[r["cls"]]
        ax.scatter(i, r[k_raw], s=MK_S, marker=MK[r["arch"]], facecolor=c, edgecolor="black", linewidth=0.4, zorder=3)
        if np.isfinite(r[k_cal]):
            ax.scatter(i, r[k_cal], s=MK_S, marker=MK[r["arch"]], facecolor="white", edgecolor=c, linewidth=1.0, zorder=4)
            ax.plot([i, i], [r[k_raw], r[k_cal]], color=c, linewidth=0.6, zorder=2)
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels([r["start"] for r in T9], rotation=90, fontsize=FS_TICK)
    ax.set_xlim(-0.7, len(T9) - 0.3)
    ax.set_ylabel("$P_2$ (nat):\nCE(reuse $S_4$)\n$-$ CE($S_2$)")
    ax.set_title(title, fontsize=FS_LABEL, loc="left"); ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
ce_panel(ax_c, "P2_ce_raw", "P2_cecal_raw", "(c) $P_2$ with stored BN statistics")
ce_panel(ax_d, "P2_ce_matched", "P2_cecal_matched", "(d) $P_2$ with BN statistics recomputed at $e=2$")
ymax = max(np.nanmax([r["P2_ce_raw"] for r in T9]), np.nanmax([r["P2_ce_matched"] for r in T9])) * 1.08
for ax in (ax_c, ax_d): ax.set_ylim(min(-0.03, np.nanmin([r["P2_cecal_raw"] for r in T9]) - 0.02), ymax)

handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=COL[c], markeredgecolor="black", markersize=MK_MS, label=CLS_LABEL[c].replace("BloodMNIST64, ", "BloodMNIST64,\n").replace(" start (", "\nstart (")) for c in CLS]
handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=MK_MS, label="SpikingResformer"),
            Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=MK_MS, label="Spikformer"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="black", markeredgecolor="black", markersize=MK_MS, label="ordinary CE (c, d)"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.2, markersize=MK_MS, label="temperature-scaled\nCE (c, d)")]
ax_c.tick_params(labelbottom=False)   # (c) and (d) share the 22 start names printed under (d)
fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
for ext in ("pdf", "png", "svg"):   # no bbox_inches="tight": the page is exactly FIGW wide
    fig.savefig(os.path.join(OUT, f"fig7_independent_starts.{ext}"), dpi=300)
    fig.savefig(os.path.join(FIGDIR, f"result_starts.{ext}"), dpi=300)
print("figure written")
if "--figure-only" in sys.argv:
    sys.exit(0)  # do not rewrite class_summary_K2048.csv or the .tex tables

# ------------------------------------------------------------------ class summary for the text
def summ(rows, k):
    a = np.array([r[k] for r in rows], float); a = a[np.isfinite(a)]
    if len(a) == 0: return dict(n=0)
    return dict(n=len(a), pos=int((a > 0).sum()), neg=int((a < 0).sum()), mean=float(a.mean()), lo=float(a.min()), hi=float(a.max()))
keys = ["P2_acc_raw", "P2_acc_matched", "P2_ce_raw", "P2_ce_matched", "P2_cecal_raw", "P2_cecal_matched", "G22_acc_raw", "G22_ce_raw",
        "G44_acc_raw", "G44_ce_raw", "G44_cecal_raw", "P4_acc_matched", "P4_ce_matched", "P4_cecal_matched", "P4_ce_raw",
        "MTT2_acc_raw", "MTT2_ce_raw", "MTT4_acc_matched", "MTT4_ce_matched", "MTT4_cecal_matched", "S0_acc_T2", "S0_acc_T4"]
cs = []
for cls in CLS + ["all"]:
    rows = T9 if cls == "all" else [r for r in T9 if r["cls"] == cls]
    for k in keys:
        d = summ(rows, k); d.update(cls=cls, column=k); cs.append(d)
with open(os.path.join(OUT, "class_summary_K2048.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["cls", "column", "n", "pos", "neg", "mean", "lo", "hi"]); w.writeheader()
    for d in cs: w.writerow({k: d.get(k, "") for k in w.fieldnames})
print("\nclass summary (K=2048):")
for cls in CLS + ["all"]:
    print(" ", cls)
    for k in ("S0_acc_T2", "P2_acc_raw", "P2_ce_raw", "P2_cecal_raw", "P2_cecal_matched", "G44_acc_raw", "G44_ce_raw", "P4_ce_matched", "P4_acc_matched"):
        d = [x for x in cs if x["cls"] == cls and x["column"] == k][0]
        if d["n"]: print(f"    {k:18s} n={d['n']:2d} +{d.get('pos',0):2d} -{d.get('neg',0):2d} mean={d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")

# ------------------------------------------------------------------ Table 9 LaTeX
def n3(v, d=3, pp=False):
    if not np.isfinite(v): return "--"
    s = f"{v:+.{d}f}" if not pp else f"{v:+.2f}"
    return s.replace("-", "$-$").replace("+", "$+$")
short = {"CIFAR-100, 30-epoch": "30-ep.", "CIFAR-100, 600-epoch": "600-ep.", "saturated, 600-epoch": "600-ep.", "PlantWild, 600-epoch": "600-ep."}
archs = {"SpikingResformer": "Resf.", "Spikformer": "Spikf."}
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Per-start contrasts at 2,048 updates for the 22 independently constructed starting checkpoints. Each entry is the mean over the adaptation seeds of that start ($n$). Accuracy contrasts are candidate minus reference in percentage points; CE contrasts are reference minus candidate in nats, so positive values favour the candidate in both cases. $P_2$ compares the two-step source with the reused four-step source at $T=2$; $G_{4,4}$ compares the four-step source with the retained start at $T=4$; $P_4^{\mathrm{BN}}$ compares the four-step source with the two-step source at $T=4$ with BN statistics recomputed at four steps. $\mathrm{CE}_{\mathrm{cal}}$ applies one validation-fitted temperature per route, using the logits saved at scoring time for the four additional datasets and the logits re-exported from the released checkpoints for the CIFAR-100 starts (Section~\ref{sec:scoring}).}")
lines.append(r"\label{tab:starts}")
lines.append(r"\begingroup\fontsize{7}{8.6}\selectfont\setlength{\tabcolsep}{3pt}")
lines.append(r"\begin{tabular}{llcrr rrrrr rr rrr}")
lines.append(r"\toprule")
lines.append(r" & & & \multicolumn{2}{c}{$S_0$ acc.\ (\%)} & \multicolumn{5}{c}{$P_2$ at $T=2$} & \multicolumn{2}{c}{$G_{4,4}$ at $T=4$} & \multicolumn{3}{c}{$P_4^{\mathrm{BN}}$ at $T=4$}\\")
lines.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-10}\cmidrule(lr){11-12}\cmidrule(lr){13-15}")
lines.append(r"Start & Dataset, arch., epochs & $n$ & $T{=}2$ & $T{=}4$ & $\Delta$acc & $\Delta\mathrm{CE}$ & $\Delta\mathrm{CE}^{\mathrm{BN}}$ & $\Delta\mathrm{CE}_{\mathrm{cal}}$ & $\Delta\mathrm{CE}^{\mathrm{BN}}_{\mathrm{cal}}$ & $\Delta$acc & $\Delta\mathrm{CE}$ & $\Delta$acc & $\Delta\mathrm{CE}$ & $\Delta\mathrm{CE}_{\mathrm{cal}}$\\")
lines.append(r"\midrule")
prev = None
for r in T9:
    if prev is not None and prev != r["cls"]:
        lines.append(r"\addlinespace[2pt]")
    prev = r["cls"]
    desc = f"{r['dataset']}, {archs[r['arch']]}, {int(r['epochs'])}"
    lines.append(" & ".join([r["start"].replace("'", r"$'$"), desc, str(int(r["n_seeds"])), f"{r['S0_acc_T2']:.1f}", f"{r['S0_acc_T4']:.1f}",
                              n3(r["P2_acc_raw"], pp=True), n3(r["P2_ce_raw"]), n3(r["P2_ce_matched"]), n3(r["P2_cecal_raw"]), n3(r["P2_cecal_matched"]),
                              n3(r["G44_acc_raw"], pp=True), n3(r["G44_ce_raw"]),
                              n3(r["P4_acc_matched"], pp=True), n3(r["P4_ce_matched"]), n3(r["P4_cecal_matched"])]) + r"\\")
lines.append(r"\midrule")
def sc(k):
    a = np.array([r[k] for r in T9], float); a = a[np.isfinite(a)]
    return f"{int((a>0).sum())}/{len(a)}"
lines.append(" & ".join(["Positive / available", "", "", "", "", sc("P2_acc_raw"), sc("P2_ce_raw"), sc("P2_ce_matched"), sc("P2_cecal_raw"), sc("P2_cecal_matched"),
                          sc("G44_acc_raw"), sc("G44_ce_raw"), sc("P4_acc_matched"), sc("P4_ce_matched"), sc("P4_cecal_matched")]) + r"\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\endgroup")
lines.append(r"\par\smallskip\begin{minipage}{.98\textwidth}\footnotesize Resf.\ denotes \texttt{spikingresformer\_cifar}; Spikf.\ denotes Spikformer-4-384. A, B, F and C are the cohorts A-expl, B-expl, F-arch and C-conf; C$'$1--3 and F$'$1--3 are the independent replication starts of Section~\ref{sec:design}; the four additional datasets use three independent 600-epoch starts each. $\Delta\mathrm{CE}^{\mathrm{BN}}$ recomputes BN statistics at the inference window for both sources. For C the four-step source is the seven-rate native selection ($1.25\times10^{-6}$); the common-grid variant is reported in the supplementary tables.\end{minipage}")
lines.append(r"\end{table*}")
tex = "\n".join(lines) + "\n"
open(os.path.join(OUT, "independent_starts_K2048.tex"), "w", encoding="utf-8").write(tex)
open(os.path.join(TABDIR, "independent_starts_K2048.tex"), "w", encoding="utf-8").write(tex)

# MTT table (14 starts)
M = [r for r in T9 if np.isfinite(r["MTT2_ce_raw"])]
ml = [r"\begin{table}[t]", r"\centering",
      r"\caption{Stage-wise MTT against the two separately adapted sources at 2,048 updates, per start. $T=2$ uses the raw stored statistics of each source; $T=4$ uses statistics recomputed at four steps. Positive accuracy favours MTT; positive CE (reference minus MTT) favours MTT.}",
      r"\label{tab:startsMTT}", r"\begingroup\fontsize{7}{8.6}\selectfont\setlength{\tabcolsep}{3.5pt}",
      r"\begin{tabular}{lrr rrr}", r"\toprule",
      r" & \multicolumn{2}{c}{$T=2$ vs. $S_2$, raw} & \multicolumn{3}{c}{$T=4$ vs. $S_4$, matched BN}\\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
      r"Start & $\Delta$acc & $\Delta\mathrm{CE}$ & $\Delta$acc & $\Delta\mathrm{CE}$ & $\Delta\mathrm{CE}_{\mathrm{cal}}$\\", r"\midrule"]
for r in M:
    ml.append(" & ".join([r["start"], n3(r["MTT2_acc_raw"], pp=True), n3(r["MTT2_ce_raw"]), n3(r["MTT4_acc_matched"], pp=True), n3(r["MTT4_ce_matched"]), n3(r["MTT4_cecal_matched"])]) + r"\\")
ml.append(r"\midrule")
def scm(k):
    a = np.array([r[k] for r in M], float); a = a[np.isfinite(a)]; return f"{int((a>0).sum())}/{len(a)}"
ml.append(" & ".join(["Positive / available", scm("MTT2_acc_raw"), scm("MTT2_ce_raw"), scm("MTT4_acc_matched"), scm("MTT4_ce_matched"), scm("MTT4_cecal_matched")]) + r"\\")
ml += [r"\bottomrule", r"\end{tabular}", r"\endgroup",
       r"\par\smallskip\begin{minipage}{.98\linewidth}\footnotesize A uses the A-conf seeds 10021--10023 (MTT at $8\times10^{-5}$ against the A-conf native pair); C uses the common-grid pair of Section~\ref{app:planned}; the four additional datasets use the MTT recipe of Section~\ref{sec:design}. B, F and the replication starts have no MTT trajectory.\end{minipage}", r"\end{table}"]
mtex = "\n".join(ml) + "\n"
open(os.path.join(OUT, "independent_starts_MTT_K2048.tex"), "w", encoding="utf-8").write(mtex)
open(os.path.join(TABDIR, "independent_starts_MTT_K2048.tex"), "w", encoding="utf-8").write(mtex)
print("tables written")
