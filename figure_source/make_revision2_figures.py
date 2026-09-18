import os
"""Publication figures from the completed, frozen R2.1 analysis tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'pgf.texsystem':'pdflatex', 'pgf.rcfonts':False,
    'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amssymb}',
    'axes.unicode_minus':False})

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'support/revision2_data';FIG=ROOT/'figures'
COL={2:'#176B91',4:'#C46A20'}
NAMES={'language':'SmoothSpike / WikiText-2','vision':'SpikingResformer / CIFAR-100'}
FINAL={'language':4096,'vision':2048}


def finish(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',pad_inches=.05,dpi=220)
    if os.environ.get('SNN3_EXPORT_PGF') == '1': fig.savefig(FIG/f'{name}.pgf',bbox_inches='tight',pad_inches=.05)
    plt.close(fig)


def main():
    check=json.loads((DATA/'verification.json').read_text(encoding='utf-8'))
    assert check['status']=='passed' and check['scoring_paths']==126 and check['timing_calls']==2880
    scores=pd.read_csv(DATA/'quality_scores.csv')
    pairs=pd.read_csv(DATA/'paired_contrasts.csv')
    intervals=pd.read_csv(DATA/'contrast_intervals.csv')
    times=pd.read_csv(DATA/'timing_checkpoint_summary.csv')
    assert len(scores)==126 and len(pairs)==26 and len(times)==32
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

    # Fixed-column differences distinguish target adaptation from state reuse.
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.1),layout='constrained')
    for j,task in enumerate(['language','vision']):
        ax=axes[j];sub=pairs[(pairs.task==task)&(pairs.calibration=='raw')]
        groups=[(512,2),(512,4),(FINAL[task],2),(FINAL[task],4)]
        for y,(K,T) in zip([3,2,1,0],groups):
            vals=sub[sub.updates==K].sort_values('seed')[f'P{T}'].to_numpy()
            row=intervals[(intervals.task==task)&(intervals.calibration=='raw')&
                (intervals.updates.astype(str)==str(K))&(intervals.contrast==f'P{T}')].iloc[0]
            ax.scatter(vals,y+np.linspace(-.13,.13,len(vals)),s=14,c=COL[T],alpha=.6,zorder=3)
            ax.errorbar(row['mean'],y,xerr=[[row['mean']-row.ci95_low],[row.ci95_high-row['mean']]],
                fmt='s',ms=4,c=COL[T],lw=1.5,capsize=3,zorder=4)
        ax.axvline(0,c='#666',ls='--',lw=.8)
        ax.axhline(1.5,c='#D9DDE0',lw=.7)
        ax.set(yticks=[3,2,1,0],yticklabels=['512: $P_2$','512: $P_4$',f'{FINAL[task]:,}: $P_2$',f'{FINAL[task]:,}: $P_4$'],
            xlabel='Cross-entropy: reused minus target-adapted',title=f'({chr(97+j)}) {NAMES[task]}',ylim=(-.48,3.48))
        ax.grid(axis='x',alpha=.18,lw=.6);ax.set_axisbelow(True)
    finish(fig,'revision2_target_effects')

    # Retain the existing endpoint table, including its row ordering.
    joined=[]
    for task in ['language','vision']:
        native=scores[(scores.task==task)&(scores.state=='adapted')&(scores.updates==FINAL[task])&
            (scores.calibration=='raw')&(scores.adapt_T==scores.infer_T)]
        for engine in ['eager','graph']:
            for seed,g in native[native.engine==engine].groupby('seed'):
                for r in g.sort_values('infer_T').itertuples(index=False):
                    T=int(r.infer_T)
                    tm=times[(times.task==task)&(times.seed==seed)&(times['T']==T)&(times.engine==engine)].iloc[0]
                    joined.append({'task':task,'seed':int(seed),'T':T,'engine':engine,'ce':r.ce,'accuracy':r.accuracy,
                        'median_of_process_medians_ms':tm['median'],'process_min_ms':tm['min'],'process_max_ms':tm['max']})
    pd.DataFrame(joined).to_csv(DATA/'paper_quality_latency.csv',index=False)

    from revision2_panels import redraw
    redesign = redraw(DATA, finish)

    groups=['task','state','updates','adapt_T','infer_T','calibration','engine']
    summary=scores.groupby(groups,dropna=False).agg(n=('ce','size'),ce_mean=('ce','mean'),ce_sd=('ce','std'),
        accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std')).reset_index()
    summary.to_csv(DATA/'paper_quality_summary.csv',index=False)
    audit={'redesign':redesign,'source_analysis':check,'generated_figures':['revision2_target_effects','revision2_bn_effects','revision2_quality_latency'],
        'manuscript_roles':{'revision2_target_effects':'Figure 2','revision2_bn_effects':'Figure 3','revision2_quality_latency':'Figure S4'},
        'quality_latency_points':len(joined),'quality_source':'actual engine, full fixed test, batch four',
        'quality_uncertainty':'paired training seeds; sample SD or Student 95% intervals as labeled',
        'timing_uncertainty':'all three process medians per checkpoint, plus their median; no pooling of calls as models',
        'BN':'all six state/inference cells at each budget processed symmetrically',
        'no_omitted_rows':len(scores)==126,'no_synthetic_result_data':True}
    (DATA/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':main()
