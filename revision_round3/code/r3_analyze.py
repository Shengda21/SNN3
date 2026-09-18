"""Recompute every R3 score and its prespecified paired comparisons."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

WORK=Path(os.environ.get('R2_WORK',Path(__file__).resolve().parents[1]))
OUT=WORK/'analysis'
def interval(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean())
    sd=float(x.std(ddof=1)) if n>1 else None
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return {'n':n,'mean':mean,'sd':sd,'low':mean-half if half is not None else None,
            'high':mean+half if half is not None else None,'positive':int((x>0).sum()),'negative':int((x<0).sum())}
def main():
    OUT.mkdir(exist_ok=True)
    queue=json.loads((WORK/'protocol/scoring_queue.json').read_text());records=[];legacy=[]
    for r in queue:
        path=WORK/'results/scoring'/r['id']
        meta=json.loads((path/'complete.json').read_text());a=np.load(path/'test.npz')
        assert meta['status']=='complete' and np.isfinite(a['loss_sum']).all()
        count=int(a['counts'].sum());ce=float(a['loss_sum'].sum()/count);acc=float(a['correct'].sum()/count)
        assert abs(ce-meta['ce'])<1e-12 and abs(acc-meta['accuracy'])<1e-12
        assert int((a['pred']==a['labels']).sum())==int(a['correct'].sum())
        assert count==(40980 if r['task']=='language' else 10000)
        records.append({**r,'ce':ce,'accuracy':acc,'actual_engine':meta['engine'],'targets':count})
        if r['role']=='legacy_replay':legacy.append({'id':r['id'],**meta['legacy_comparison']})
    scores=pd.DataFrame(records);scores.to_csv(OUT/'quality_scores.csv',index=False)
    pd.DataFrame(legacy).to_csv(OUT/'legacy_runtime_comparison.csv',index=False)
    primary=scores[scores.role=='primary'].copy()
    for init,expected in [('A',set(range(9621,9626))),('B',set(range(9721,9724)))]:
        group=primary[(primary.init_id==init)&(primary.state=='adapted')]
        assert set(group.seed.astype(int))==expected
        for _,part in group.groupby(['updates','adapt_T','infer_T','calibration','cal_T','cal_seed'],dropna=False):
            assert set(part.seed.astype(int))==expected
    assert len(legacy)==126
    primary['condition']=np.where(primary.calibration=='raw','raw','bn')
    diag=primary[(primary.calibration=='raw')|((primary.cal_T==primary.infer_T)&(primary.cal_seed==9640))]
    detail=[]
    for init in ['A','B']:
        d=diag[diag.init_id==init]
        for cond in ['raw','bn']:
            q=d[d.condition==cond]
            for e in [2,4]:
                base=q[(q.state=='shared_initial')&(q.infer_T==e)].iloc[0]
                for K in [512,1024,2048]:
                    block=q[(q.updates==K)&(q.infer_T==e)]
                    for seed,g in block.groupby('seed'):
                        g=g.set_index('adapt_T');own=g.loc[e];other=g.loc[6-e]
                        common={'init_id':init,'condition':cond,'infer_T':e,'updates':K,'seed':int(seed)}
                        detail.append({**common,'contrast':'P','value':float(other.ce-own.ce)})
                        detail.append({**common,'contrast':'accuracy_target_minus_reuse_pp','value':float(100*(own.accuracy-other.accuracy))})
                        for a in [2,4]:
                            detail.append({**common,'contrast':f'G_a{a}','value':float(base.ce-g.loc[a].ce)})
                            detail.append({**common,'contrast':f'accuracy_gain_a{a}_pp','value':float(100*(g.loc[a].accuracy-base.accuracy))})
    effects=pd.DataFrame(detail);effects.to_csv(OUT/'contrasts_by_seed.csv',index=False)
    summaries=[]
    keys=['init_id','condition','infer_T','updates','contrast']
    for key,g in effects.groupby(keys):summaries.append({**dict(zip(keys,key)),**interval(g.value)})
    pd.DataFrame(summaries).to_csv(OUT/'contrast_intervals.csv',index=False)
    interactions=[]
    for init,g in effects[(effects.contrast=='P')&(effects.infer_T==4)].groupby('init_id'):
        for seed,h in g.groupby('seed'):
            h=h.set_index(['condition','updates'])
            raw=h.loc[('raw',2048),'value']-h.loc[('raw',512),'value']
            bn=h.loc[('bn',2048),'value']-h.loc[('bn',512),'value']
            interactions.append({'init_id':init,'seed':int(seed),'raw_change':raw,'bn_change':bn,'J':bn-raw})
    ji=pd.DataFrame(interactions);ji.to_csv(OUT/'interaction_by_seed.csv',index=False)
    js=[{'init_id':key,**interval(g.J)} for key,g in ji.groupby('init_id')]
    pd.DataFrame(js).to_csv(OUT/'interaction_intervals.csv',index=False)
    # A positive transfer penalty favors statistics recalibrated at the inference window.
    penalties=[]
    final=primary[(primary.updates.isin([0,2048]))&(primary.calibration!='raw')]
    for key,g in final.groupby(['init_id','source_id','cal_seed','infer_T']):
        init,source,subset,e=key;g=g.set_index('cal_T');matched=g.loc[e];shared=g.loc[6-e]
        penalties.append({'init_id':init,'source_id':source,'state':matched.state,'seed':matched.seed,
            'adapt_T':matched.adapt_T,'cal_seed':int(subset),'infer_T':int(e),
            'CE_other_cal_minus_matched':float(shared.ce-matched.ce),
            'accuracy_matched_minus_other_pp':float(100*(matched.accuracy-shared.accuracy))})
    penalties=pd.DataFrame(penalties);penalties.to_csv(OUT/'bn_transfer_penalty_by_seed.csv',index=False)
    ps=[]
    for key,g in penalties.groupby(['init_id','state','adapt_T','cal_seed','infer_T'],dropna=False):
        fields=dict(zip(['init_id','state','adapt_T','cal_seed','infer_T'],key))
        for metric in ['CE_other_cal_minus_matched','accuracy_matched_minus_other_pp']:
            ps.append({**fields,'metric':metric,**interval(g[metric])})
    pd.DataFrame(ps).to_csv(OUT/'bn_transfer_penalty_intervals.csv',index=False)
    # The common-LR source may already be a primary source; do not duplicate scoring.
    common=scores[(scores.init_id=='A')&(scores.updates==2048)&(scores.lr==2e-5)]
    common=common[(common.calibration=='raw')|((common.cal_T==common.infer_T)&(common.cal_seed==9640))]
    common_effects=[]
    for (cal,e),g in common.groupby(['calibration','infer_T']):
        for seed,h in g.groupby('seed'):
            h=h.set_index('adapt_T');assert len(h)==2
            common_effects.append({'seed':int(seed),'condition':'raw' if cal=='raw' else 'bn','infer_T':int(e),
                                  'P':float(h.loc[6-e].ce-h.loc[e].ce)})
    common_effects=pd.DataFrame(common_effects);common_effects.to_csv(OUT/'common_lr_by_seed.csv',index=False)
    cs=[]
    for (cal,e),g in common_effects.groupby(['condition','infer_T']):cs.append({'condition':cal,'infer_T':int(e),**interval(g.P)})
    pd.DataFrame(cs).to_csv(OUT/'common_lr_intervals.csv',index=False)
    group=['init_id','state','updates','adapt_T','infer_T','calibration','cal_T','cal_seed']
    means=primary.groupby(group,dropna=False).agg(n=('ce','size'),ce_mean=('ce','mean'),ce_sd=('ce','std'),accuracy_mean=('accuracy','mean')).reset_index()
    means.to_csv(OUT/'quality_mean_matrix.csv',index=False)
    summary={'status':'passed','scoring_paths':len(records),'legacy_paths':len(legacy),'new_primary_paths':len(primary),
        'initializations':2,'adaptation_seed_counts':{'A':5,'B':3},'calibration_subsets':3,
        'calibration_subset_inflates_training_n':False,'interval_scope':'pointwise Student across adaptation seeds, conditional on initialization and calibration subset',
        'J':js,'actual_engines':scores.actual_engine.value_counts().to_dict(),
        'legacy_max_absolute_mean_ce_shift':float(max(abs(x['mean_ce_change']) for x in legacy)),
        'legacy_prediction_changes_total':sum(x['prediction_changes'] for x in legacy)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
