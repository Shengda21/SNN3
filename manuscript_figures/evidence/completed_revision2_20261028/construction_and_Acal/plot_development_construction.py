"""Render two supplied development construction histories; no new model/data work."""
import csv
import hashlib
import json
import sys
import math
from pathlib import Path

import matplotlib
import matplotlib._mathtext as _mt; _mt.SHRINK_FACTOR = 0.92
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Figure-only by default: tables/CSV/JSON/caption evidence files are rewritten only with --write-evidence.
WRITE_EVIDENCE='--write-evidence' in sys.argv
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'existing_eight_construction_curves_20261027.csv'
raw_rows=list(csv.DictReader(SOURCE.open(encoding='utf-8')))
rows=[row for row in raw_rows if row['role']=='development'
      and (row['cohort'],row['construction_seed']) in (('C-conf','12650'),('F-arch','12750'))]
assert len(rows)==630
groups={cohort:sorted([row for row in rows if row['cohort']==cohort],key=lambda row:int(row['epoch']))
        for cohort in ('C-conf','F-arch')}
assert [int(row['epoch']) for row in groups['C-conf']]==list(range(1,601))
assert [int(row['epoch']) for row in groups['F-arch']]==list(range(1,31))
selected={cohort:next(row for row in group if row['selected']=='True') for cohort,group in groups.items()}
assert [int(selected[c]['epoch']) for c in groups]==[556,29]
assert all(sum(row['selected']=='True' for row in group)==1 for group in groups.values())

subset=HERE/'development_C12650_F12750_source_630rows.csv'
if WRITE_EVIDENCE:
    with subset.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw_rows[0]));writer.writeheader();writer.writerows(rows)

# Visual contract: supplied per-epoch validation observations only; no smoothing.
# Single-column layout (3.3 in = \columnwidth, all text >= 8 pt): four panels stacked
# vertically, cohort by cohort (600-/30-epoch histories), accuracy above CE.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,
    'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
    'xtick.major.width':.65,'ytick.major.width':.65,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.facecolor':'white'})
blue='#0072B2';orange='#D55E00'
fig,axes=plt.subplots(4,1,figsize=(3.3,6.9),layout='constrained')
fig.get_layout_engine().set(h_pad=.05,w_pad=.03,hspace=.04)
ce_max=max(float(row['construction_validation_ce']) for row in rows)
ce_limit=math.ceil(ce_max*2)/2
panel=iter('abcd')
for i,(cohort,seed,epochs) in enumerate((('C-conf',12650,600),('F-arch',12750,30))):
    group=groups[cohort];choice=selected[cohort]
    epoch=[int(row['epoch']) for row in group];selected_epoch=int(choice['epoch'])
    for j,(field,label,factor) in enumerate((('construction_validation_accuracy','Validation accuracy (%)',100.),
                                           ('construction_validation_ce','Validation CE',1.))):
        ax=axes[2*i+j];y=[float(row[field])*factor for row in group]
        selected_y=float(choice[field])*factor
        ax.plot(epoch,y,color=blue,lw=1.)
        ax.axvline(selected_epoch,color='#777777',ls=(0,(3,2)),lw=.7,zorder=1)
        ax.plot(selected_epoch,selected_y,'o',color=orange,markersize=4,zorder=3)
        ax.annotate('e'+str(selected_epoch),xy=(selected_epoch,selected_y),
            xytext=(-30,-20 if j==0 else 20),textcoords='offset points',fontsize=8,
            ha='right',va='center',color='#333333',
            arrowprops={'arrowstyle':'-','lw':.6,'color':'#777777'})
        ax.set_title('('+next(panel)+') '+cohort, loc='left', pad=5)
        ax.set_xlabel('Construction epoch');ax.set_ylabel(label)
        ax.set_xlim(0,epochs)
        ax.set_ylim((0,85 if cohort=='C-conf' else 70) if j==0 else (0,ce_limit))
        ax.grid(axis='y',color='#DEDEDE',lw=.45,zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=3,pad=2)
fig.legend([Line2D([0],[0],color=blue,lw=1),Line2D([0],[0],marker='o',ls='None',color=orange,markersize=4)],
    ['Construction validation','Selected checkpoint'],ncol=2,loc='outside upper center',frameon=False,
    handlelength=1.4,columnspacing=1.0,handletextpad=.5)
prefix=HERE/'development_construction_validation'
for extension in ('pdf','svg','png'):
    fig.savefig(prefix.with_suffix('.'+extension),dpi=300)
plt.close(fig)

if WRITE_EVIDENCE:
    caption=('Construction-validation histories for development checkpoints C12650 (600 epochs) and '
        'F12750 (30 epochs). Unsmoothed per-epoch accuracy and CE are shown; markers identify the '
        'selected checkpoints at epochs556 and29. Construction used AMP training with TF32 allowed, '
        'while validation evaluation was outside autocast. These values are distinct from the '
        'native FP32, TF32-disabled test evaluations. Source: the630 development rows of '
        'existing_eight_construction_curves_20261027.csv.')
    (HERE/'construction_figure_caption.txt').write_text(caption+'\n',encoding='utf-8')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    metadata={'status':'supplied_curves_rendered_no_new_evaluation','source':{'path':str(SOURCE.relative_to(HERE.parent.parent)),
        'sha256':sha(SOURCE),'rows_in_full_source':len(raw_rows)},'selection':{'role':'development','construction_seeds':[12650,12750]},
        'plotted_rows':630,'source_CSV_header_extra_to_630_data_rows':True,'epoch_counts':{'C-conf':600,'F-arch':30},
        'selected_rows':selected,'axes':{'rows':['C-conf','F-arch'],'columns':['validation accuracy percent','validation CE'],
        'accuracy_transformation':'source fraction multiplied by100','smoothing':None},
        'precision_scope':'AMP training, TF32 allowed; construction validation outside autocast; not native FP32 test',
        'base_font_pt':9,'figure_inches':[7.1,4.7],'SVG_text_editable':True,'new_GPU_or_model_or_dataset_work':False,
        'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in
            [subset,*[prefix.with_suffix('.'+ext) for ext in ('pdf','svg','png')]]}}
    (HERE/'construction_figure_sources.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
assert '<text' in prefix.with_suffix('.svg').read_text() and '<image' not in prefix.with_suffix('.svg').read_text()
print(json.dumps({'status':'supplied_curves_rendered_no_new_evaluation','plotted_rows':630,'selected_epochs':[556,29],
    'files':[p.name for p in [subset,*[prefix.with_suffix('.'+ext) for ext in ('pdf','svg','png')]]]}))
