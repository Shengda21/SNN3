import os
"""All numerical figures are derived from preserved experimental outputs."""
from pathlib import Path
import json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'pgf.texsystem':'pdflatex', 'pgf.rcfonts':False,
    'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amssymb}',
    'axes.unicode_minus':False})
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
DATA=ROOT/'support/figure_data';DATA.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
ratios=pd.read_csv(DATA/'execution_ratios.csv')
diffs=read(DATA/'quality_differences.json')
COL={2:'#176B91',4:'#C46A20'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
 'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def finish(fig,name):
 fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
 fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.03)
 if os.environ.get('SNN3_EXPORT_PGF') == '1': fig.savefig(FIG/f'{name}.pgf',bbox_inches='tight',pad_inches=.05)
 plt.close(fig)
def grid(ax):ax.grid(axis='y',alpha=.20,linewidth=.6);ax.set_axisbelow(True)

# Original official-test outputs, with a post-review descriptive recovery view.
q=pd.read_csv(DATA/'quality_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(7.0,2.7),layout='constrained')
for T in [2,4]:
 z=q[(q.stage=='formal')&(q['T']==T)].sort_values('step')
 initial=float(q[(q.stage=='frozen')&(q['T']==T)].ce_mean.iloc[0])
 axes[0].errorbar(np.r_[0,z.step],np.r_[initial,z.ce_mean],yerr=np.r_[0,z.ce_sd],marker='o',markersize=4,color=COL[T],label=f'T = {T}',capsize=2)
axes[0].set(xlabel='Adaptation updates',ylabel='Test cross-entropy',title='(a) Quality along adaptation',xticks=[0,512,1024,2048,4096],ylim=(2.51,3.25))
axes[0].tick_params(axis='x',rotation=25);axes[0].legend(loc='upper right',frameon=False)
frozen_gap=float(q[(q.stage=='frozen')&(q['T']==2)].ce_mean.iloc[0]-q[(q.stage=='frozen')&(q['T']==4)].ce_mean.iloc[0])
ds=sorted([d for d in diffs if d['split']=='test' and d['stage']=='formal' and d['comparison']=='matched' and d.get('step') in [512,1024,2048,4096]],key=lambda d:d['step'])
x=np.r_[0,[d['step'] for d in ds]]
gaps=np.array([d['ce_difference_each_seed'] for d in ds])
recovered=np.vstack([np.zeros(3),100*(frozen_gap-gaps)/frozen_gap])
for j,(seed,marker) in enumerate(zip([8621,8622,8623],['o','s','^'])):
 axes[1].plot(x,recovered[:,j],c=['#657987','#989277','#A2878C'][j],marker=marker,ms=3.4,lw=.8,alpha=.7,label=str(seed))
axes[1].plot(x,recovered.mean(axis=1),c=COL[2],lw=1.7,label='Mean')
axes[1].set(xlabel='Adaptation updates',ylabel='Initial loss gap recovered (%)',title='(b) Recovery relative to initialization',xticks=[0,512,1024,2048,4096],ylim=(-1,50))
axes[1].tick_params(axis='x',rotation=25);axes[1].legend(loc='lower right',frameon=False,fontsize=8,ncol=2)
for ax in axes:grid(ax)
finish(fig,'adaptation')

# Every independent session or training seed is drawn, including the outlier.
fig,axes=plt.subplots(2,3,figsize=(7.05,4.6),sharex=True,layout='constrained')
groups=[('historical','language','Frozen language'),('historical','vision','Reconstructed vision'),('formal','language','Adapted language')]
ratio_cols=['same_T4_graph_over_T2_graph','cross_T2_eager_over_T4_graph']
for j,(g,arch,title) in enumerate(groups):
 sub=ratios[(ratios.group==g)&(ratios.architecture==arch)&(ratios.route=='full')]
 for i,col in enumerate(ratio_cols):
  ax=axes[i,j]
  for k,(session,z) in enumerate(sub.groupby('session')):
   z=z.sort_values('batch');ax.plot(z.batch,z[col],marker=['o','s','^'][k],ms=4,lw=.9,c=[COL[2],COL[4],'#57665E'][k],alpha=.85)
  ax.axhline(1,c='#555',lw=.8,ls='--');ax.set_xscale('log',base=4);ax.set_xticks([1,4,16,64],['1','4','16','64']);grid(ax)
  if i==0:ax.set_title(f'({chr(97+j)}) {title}');ax.set_ylim(1.2,2.2)
  else:ax.set_ylim(.25,4.7);ax.set_xlabel('Batch size')
axes[0,0].set_ylabel('T4 graph / T2 graph')
axes[1,0].set_ylabel('T2 eager / T4 graph')
finish(fig,'execution')

# Primary joint evidence shares the quality-scoring engine and batch size.
p=pd.read_csv(DATA/'frontier.csv')
fig,axes=plt.subplots(1,2,figsize=(7.0,2.9),sharey=True,layout='constrained')
for ax,engine,title in zip(axes,['eager','graph'],['(a) Eager: scoring protocol','(b) Graph: reference quality']):
 z=p[(p.batch==4)&(p.engine==engine)]
 for k,seed in enumerate([8621,8622,8623]):
  pair=z[z.seed==seed].sort_values('T')
  ax.plot(pair.latency_ms,pair.ce,c='#859098',lw=.8,alpha=.6,zorder=1)
  for T in [2,4]:
   v=pair[pair['T']==T]
   ax.scatter(v.latency_ms,v.ce,s=38,marker=['o','s','^'][k],edgecolor=COL[T],facecolor=COL[T] if engine=='graph' else 'white',lw=1,zorder=3)
 ax.set(title=title,xlabel='Median batch latency (ms)',ylim=(2.53,2.70));grid(ax)
axes[0].set_ylabel('Test cross-entropy')
axes[1].set_ylabel('Reference cross-entropy')
handles=[Line2D([],[],color=COL[2],lw=2,label='T = 2'),Line2D([],[],color=COL[4],lw=2,label='T = 4')]
axes[0].legend(handles=handles,loc='upper right',frameon=False,fontsize=8)
handles=[Line2D([],[],marker=m,color='#555',ls='',label=str(seed),markerfacecolor='white') for m,seed in zip(['o','s','^'],[8621,8622,8623])]
axes[1].legend(handles=handles,loc='upper right',frameon=False,fontsize=8,title='Training seed',title_fontsize=8)
finish(fig,'frontier')

# Same checkpoints, same seed, same GPU. No averaging across training seeds.
p=pd.read_csv(DATA/'frontier.csv')
fig,axes=plt.subplots(2,2,figsize=(7.0,4.8),sharey=True,layout='constrained')
for ax,b in zip(axes.ravel(),[1,4,16,64]):
 z=p[p.batch==b]
 for k,seed in enumerate([8621,8622,8623]):
  zs=z[z.seed==seed]
  for T in [2,4]:
   t=zs[zs['T']==T].sort_values('latency_ms')
   ax.plot(t.latency_ms,t.ce,color=COL[T],alpha=.26,lw=.8)
   for e,filled in [('eager',False),('graph',True)]:
    a=t[t.engine==e];ax.scatter(a.latency_ms,a.ce,s=30,marker=['o','s','^'][k],edgecolor=COL[T],facecolor=COL[T] if filled else 'white',linewidth=.9,zorder=3)
  f=zs[zs.engine=='graph'].sort_values('latency_ms');ax.plot(f.latency_ms,f.ce,color='#777',lw=.65,ls=':',alpha=.7)
 ax.set_title(f'Batch {b}');ax.set_xlabel('Median latency per batch (ms)');grid(ax)
axes[0,0].set_ylabel('Reference test cross-entropy');axes[1,0].set_ylabel('Reference test cross-entropy')
legend=[Line2D([],[],color=COL[2],lw=2,label='T = 2'),Line2D([],[],color=COL[4],lw=2,label='T = 4'),
 Line2D([],[],marker='o',ls='',markerfacecolor='white',color='#555',label='Eager'),Line2D([],[],marker='o',ls='',color='#555',label='Graph')]
axes[0,0].legend(handles=legend,loc='center right',ncol=2,frameon=False,fontsize=8)
finish(fig,'reference_grid')

# Input copy control and required output control.
fig,axes=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
for arch,color,offset in [('language',COL[2],-.08),('vision',COL[4],.08)]:
 z=ratios[(ratios.group=='historical')&(ratios.architecture==arch)&(ratios.batch.isin([1,4]))&(ratios.route=='full')]
 for k,(session,s) in enumerate(z.groupby('session')):
  s=s.sort_values('batch');axes[0].plot(np.array([0,1])+offset,s.matched_copy_T2_eager_over_T4_graph,marker=['o','s','^'][k],ms=4,color=color,lw=.9,alpha=.8)
axes[0].set(xticks=[0,1],xticklabels=['1','4'],xlabel='Batch size',ylabel='T2 eager+copy / T4 graph+copy',title='(a) Matched device input copy',ylim=(.8,5))
axes[0].legend(handles=[Line2D([],[],color=COL[2],label='Language'),Line2D([],[],color=COL[4],label='Vision')],frameon=False)
for route,c,marker in [('full',COL[2],'o'),('masked',COL[4],'s')]:
 z=ratios[(ratios.group=='masked')&(ratios.route==route)].sort_values('batch')
 axes[1].plot(z.batch,z.cross_T2_eager_over_T4_graph,c=c,marker=marker,ms=4,lw=1,label=f'{route.capitalize()} head')
axes[1].set_xscale('log',base=4);axes[1].set_xticks([1,4,16,64],['1','4','16','64']);axes[1].set(xlabel='Batch size',ylabel='T2 eager / T4 graph',title='(b) Required masked positions')
axes[1].legend(frameon=False)
for ax in axes:ax.axhline(1,c='#555',lw=.8,ls='--');grid(ax)
finish(fig,'controls')

# Complete search, including nonselected candidates; train holdout only.
search=pd.read_csv(DATA/'learning_rate_search.csv')
fig,axes=plt.subplots(1,2,figsize=(7.0,2.7),layout='constrained',sharey=True)
for ax,T in zip(axes,[2,4]):
 for lr,c in zip(sorted(search.lr.unique()),['#6E7781','#176B91','#C46A20','#865E9A']):
  z=search[(search['T']==T)&(search.lr==lr)].groupby('step').ce.agg(['mean','min','max'])
  ax.plot(z.index,z['mean'],color=c,marker='o',ms=3,lw=1,label=f'{lr:g}')
  ax.fill_between(z.index,z['min'],z['max'],color=c,alpha=.12)
 ax.set(title=f'T = {T}',xlabel='Screening updates',xticks=[512,1024,2048]);grid(ax)
axes[0].set_ylabel('Training holdout cross-entropy');axes[1].legend(title='Learning rate',frameon=False,fontsize=8,title_fontsize=8)
finish(fig,'search')

# Observed runtime variations and preselected time checkpoints.
fig,axes=plt.subplots(1,2,figsize=(7.0,2.9),layout='constrained')
runtime=pd.read_csv(DATA/'training_runtime_chunks.csv')
for k,seed in enumerate([8621,8622,8623]):
 for T in [2,4]:
  chunks=runtime[(runtime['T']==T)&(runtime.seed==seed)].sort_values('midpoint_step')
  axes[0].plot(chunks.midpoint_step,chunks.median_seconds,color=COL[T],ls=['-','--',':'][k],lw=1)
axes[0].set(xlabel='Adaptation updates',ylabel='Median seconds per update',title='(a) Runtime changes within runs',ylim=(.38,.69));grid(axes[0])
points=pd.read_csv(DATA/'time_budget_points.csv')
for T,c,offset in [(2,COL[2],-.05),(4,COL[4],.05)]:
 for k,seed in enumerate([8621,8622,8623]):
  z=points[(points['T']==T)&(points.seed==seed)].sort_values('fraction')
  axes[1].scatter(np.arange(3)+offset,z.slack_fraction*100,c=c,marker=['o','s','^'][k],s=26)
axes[1].axhline(10,c='#555',ls='--',lw=.8);axes[1].set(xticks=[0,1,2],xticklabels=['h/4','h/2','h'],xlabel='Nominal training time cap',ylabel='Unused time fraction (%)',title='(b) Checkpoint slack');grid(axes[1])
axes[0].legend(handles=[Line2D([],[],c=COL[2],label='T = 2'),Line2D([],[],c=COL[4],label='T = 4')],loc='upper right',frameon=False)
finish(fig,'time_budget')
print('Created seven numerical figures from the preserved source tables.')
