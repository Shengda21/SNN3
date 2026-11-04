"""Absolute CE trajectories for the retained A/B adaptation cohorts.

Source Data: evidence/budget_figure/quality_by_seed.csv, a lossless selection
of r3/analysis/quality_scores.csv (raw or matched BN, calibration seed 9640,
initial states and budgets 512/1024/2048). No model execution is needed.
Each interval describes variation across adaptation seeds at one budget;
the three saved budgets repeat observations of the same trajectories.
Layout: single-column figure (3.3 in, the column width) with the four panels
stacked vertically; every text element, subscripts included, is >= 7.5 pt in
the PDF. The inference window is written $e$ in titles and labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._mathtext as _mt; _mt.SHRINK_FACTOR = 0.95; _mt.FontConstantsBase.sub1 = 0.42; _mt.FontConstantsBase.sub2 = 0.6; _mt.DejaVuSansFontConstants.sub1 = 0.42; _mt.DejaVuSansFontConstants.sub2 = 0.6
from matplotlib.lines import Line2D

SRC = Path(__file__).resolve().parents[1]
DATA = SRC.parent / 'evidence/budget_figure/quality_by_seed.csv'
OUT = SRC / 'figures'
OUT.mkdir(exist_ok=True)
d = pd.read_csv(DATA)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .65,
    'lines.linewidth': 1.25, 'savefig.facecolor': 'white'})
colors = {2: '#0072B2', 4: '#D55E00'}
budgets = np.array([512,1024,2048])
fig, axes = plt.subplots(4,1,figsize=(3.3,7.3),sharex=True,layout='constrained')
for row,init in enumerate(['A','B']):
    n = 5 if init=='A' else 3
    for col,infer in enumerate([2,4]):
        ax = axes[row*2+col]
        for adapt,norm,marker,style in [(2,'raw','o','-'),(4,'raw','^','-'),
            (2,'matched_BN','s','--'),(4,'matched_BN','D','--')]:
            q=d[(d.init_id==init)&(d.state=='adapted')&(d.adapt_T==adapt)&
                (d.infer_T==infer)&(d.normalization==norm)]
            means=[];halves=[];seedsets=[]
            for budget in budgets:
                x=q[q.updates==budget]
                assert len(x)==n and x.seed.nunique()==n
                seedsets.append(set(x.seed))
                means.append(x.ce.mean())
                halves.append(t.ppf(.975,n-1)*x.ce.std(ddof=1)/np.sqrt(n))
            assert seedsets[0]==seedsets[1]==seedsets[2]
            ax.errorbar(budgets,means,yerr=halves,color=colors[adapt],
                marker=marker,linestyle=style,ms=3.5,capsize=2,zorder=3)
        for norm,color,style in [('raw','#555555',':'),('matched_BN','#888888','-.')]:
            q=d[(d.init_id==init)&(d.state=='shared_initial')&
                (d.infer_T==infer)&(d.normalization==norm)]
            assert len(q)==1
            ax.axhline(q.ce.iloc[0],color=color,ls=style,lw=1.2,zorder=2)
        ax.set_xscale('log',base=2)
        ax.set_xticks(budgets,[str(x) for x in budgets])
        ax.set_title(f'({chr(97+row*2+col)}) {init}-expl ({n} seeds), inference $e={infer}$')
        ax.set_ylabel('Cross-entropy')
        if row*2+col==3:ax.set_xlabel('Adaptation updates')
        ax.grid(axis='y',color='#DDDDDD',lw=.5,zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=.08,y=.12)
        ax.set_xlim(450,2350)
handles=[]
for adapt,norm,marker,style in [(2,'raw','o','-'),(4,'raw','^','-'),
    (2,'matched_BN','s','--'),(4,'matched_BN','D','--')]:
    handles.append(Line2D([],[],color=colors[adapt],marker=marker,ls=style,ms=3.5,
        label=f'$S_{adapt}$, '+('raw' if norm=='raw' else 'matched BN')))
handles += [Line2D([],[],color='#555555',ls=':',label='$S_0$, raw'),
    Line2D([],[],color='#888888',ls='-.',label='$S_0$, matched BN')]
handles = [handles[i] for i in (0,1,4,2,3,5)]  # left column raw, right column matched BN
fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,
    columnspacing=1.2,handlelength=2.2)
fig.get_layout_engine().set(h_pad=.05,w_pad=.03,hspace=.03)
for ext in ['pdf','svg','png']:
    fig.savefig(OUT / ('result_absolute_budget.'+ext),dpi=240)
plt.close(fig)
print('Built result_absolute_budget.pdf/svg/png from 200 source-data rows.')
