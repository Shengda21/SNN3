"""Predefined seed-paired contrasts and nested timing summaries for revision R2.1."""
from pathlib import Path
import argparse,json,math,os
import numpy as np
import pandas as pd
from scipy.stats import t


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def interval(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean())
    sd=float(x.std(ddof=1)) if n>1 else None
    half=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    return {'n_paired_seeds':n,'mean':mean,'sample_sd':sd,
        'ci95_low':mean-half if half is not None else None,'ci95_high':mean+half if half is not None else None,
        'minimum':float(x.min()),'maximum':float(x.max()),'positive_seeds':int((x>0).sum()),'negative_seeds':int((x<0).sum())}


def analyze(root,output=None):
    out=output if output is not None else root/'analysis';out.mkdir(parents=True,exist_ok=True)
    manifest=read(root/'results/model_analysis_lock.json');rows=[];arrays={}
    for row in manifest['scoring']:
        folder=root/'results/scoring'/row['id'];meta=read(folder/'complete.json')
        with np.load(folder/'test.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        expected_items,expected_targets=(4406,40980) if row['task']=='language' else (10000,10000)
        assert all(len(a[key])==expected_items for key in ['block_id','loss_sum','counts','correct'])
        assert len(np.unique(a['block_id']))==expected_items
        assert int(a['counts'].sum())==len(a['pred'])==len(a['labels'])==expected_targets
        assert int(a['correct'].sum())==int((a['pred']==a['labels']).sum())
        assert np.isfinite(a['loss_sum']).all() and (a['counts']>=0).all() and (a['correct']<=a['counts']).all()
        ce=float(a['loss_sum'].sum(dtype=np.float64)/a['counts'].sum());acc=float(a['correct'].sum()/a['counts'].sum())
        assert np.allclose([ce,acc],[meta['ce'],meta['accuracy']],atol=1e-12,rtol=0)
        rows.append({**row,'ce':ce,'accuracy':acc,'items':len(a['block_id']),'targets':int(a['counts'].sum())})
        arrays[row['id']]=a
    scores=pd.DataFrame(rows);scores.to_csv(out/'quality_scores.csv',index=False)
    pairs=[];aggregates=[];changes=[]
    for task,seeds,budgets,cals in [('language',[8621,8622,8623],[512,4096],['raw']),
                                  ('vision',list(range(9621,9626)),[512,2048],['raw','bn_only'])]:
        for cal in cals:
            sub=scores[(scores.task==task)&(scores.calibration==cal)&(scores.engine=='eager')]
            zero={e:float(sub[(sub.state=='shared_initial')&(sub.infer_T==e)].iloc[0].ce) for e in [2,4]}
            for K in budgets:
                for seed in seeds:
                    cell={(a,e):float(sub[(sub.seed==seed)&(sub.updates==K)&(sub.adapt_T==a)&(sub.infer_T==e)].iloc[0].ce)
                          for a in [2,4] for e in [2,4]}
                    P2=cell[4,2]-cell[2,2];P4=cell[2,4]-cell[4,4]
                    D0=zero[2]-zero[4];D=cell[2,2]-cell[4,4]
                    pairs.append({'task':task,'calibration':cal,'updates':K,'seed':seed,
                        **{f'Q{a}{e}':value for (a,e),value in cell.items()},'Q02':zero[2],'Q04':zero[4],
                        **{f'gain_a{a}_e{e}':zero[e]-value for (a,e),value in cell.items()},
                        'P2':P2,'P4':P4,'I':P2+P4,'inference_T2_minus_T4_after_a2':cell[2,2]-cell[2,4],
                        'inference_T2_minus_T4_after_a4':cell[4,2]-cell[4,4],
                        'initial_native_gap':D0,'native_gap':D,'gap_recovery':D0-D,
                        'gap_recovery_percent':100*(D0-D)/D0 if D0!=0 else None})
    p=pd.DataFrame(pairs);p.to_csv(out/'paired_contrasts.csv',index=False)
    measures=['P2','P4','I','native_gap','gap_recovery','gap_recovery_percent',
        'inference_T2_minus_T4_after_a2','inference_T2_minus_T4_after_a4',
        'gain_a2_e2','gain_a2_e4','gain_a4_e2','gain_a4_e4']
    for (task,cal,K),g in p.groupby(['task','calibration','updates']):
        for key in measures:aggregates.append({'task':task,'calibration':cal,'updates':int(K),'contrast':key,**interval(g[key])})
    for (task,cal,seed),g in p.groupby(['task','calibration','seed']):
        g=g.sort_values('updates');a,b=g.iloc[0],g.iloc[-1]
        changes.append({'task':task,'calibration':cal,'seed':int(seed),'early_updates':int(a.updates),'final_updates':int(b.updates),
            **{key+'_change':float(b[key]-a[key]) for key in measures}})
    pd.DataFrame(changes).to_csv(out/'paired_budget_changes.csv',index=False)
    for (task,cal),g in pd.DataFrame(changes).groupby(['task','calibration']):
        for key in measures:aggregates.append({'task':task,'calibration':cal,'updates':'final_minus_early','contrast':key,**interval(g[key+'_change'])})
    pd.DataFrame(aggregates).to_csv(out/'contrast_intervals.csv',index=False)
    bn=[]
    for (seed,K),g in p[p.task=='vision'].groupby(['seed','updates']):
        raw=g[g.calibration=='raw'].iloc[0];cal=g[g.calibration=='bn_only'].iloc[0]
        bn.append({'seed':int(seed),'updates':int(K),**{key+'_bn_minus_raw':float(cal[key]-raw[key]) for key in measures}})
    bn_frame=pd.DataFrame(bn);bn_frame.to_csv(out/'bn_contrast_changes.csv',index=False)
    bn_intervals=[]
    for K,g in bn_frame.groupby('updates'):
        for key in measures:
            bn_intervals.append({'updates':int(K),'contrast':key,**interval(g[key+'_bn_minus_raw'])})
    pd.DataFrame(bn_intervals).to_csv(out/'bn_contrast_change_intervals.csv',index=False)
    engine=[]
    for row in rows:
        if row['engine']!='graph':continue
        raw_id=row['id'].removesuffix('_graph')+'_eager'
        a,b=arrays[raw_id],arrays[row['id']]
        assert np.array_equal(a['block_id'],b['block_id']) and np.array_equal(a['labels'],b['labels'])
        eager=next(r for r in rows if r['id']==raw_id)
        engine.append({'task':row['task'],'seed':row['seed'],'T':row['infer_T'],
            'eager_ce':eager['ce'],'graph_ce':row['ce'],'ce_graph_minus_eager':row['ce']-eager['ce'],
            'eager_accuracy':eager['accuracy'],'graph_accuracy':row['accuracy'],
            'prediction_changes':int((a['pred']!=b['pred']).sum()),'predictions_compared':len(a['pred']),
            'max_per_item_loss_sum_error':float(np.max(np.abs(a['loss_sum']-b['loss_sum']))),
            'loss_arrays_bitwise_equal':bool(np.array_equal(a['loss_sum'],b['loss_sum']))})
    pd.DataFrame(engine).to_csv(out/'engine_quality_comparison.csv',index=False)
    timing=[];timing_pairs=[];process_ids={}
    for session in manifest['timing_sessions']:
        for T in [2,4]:
            meta=read(root/'results/timing'/session['id']/f'T{T}.json')
            identity=(meta['process_pid'],meta['process_session_started_at'])
            if session['id'] in process_ids:assert process_ids[session['id']]==identity
            else:process_ids[session['id']]=identity
            for e in ['eager','graph']:
                calls=[c['wall_ms'] for c in meta['measurements'] if c['engine']==e]
                assert len(calls)==30
                q1,med,q3=np.quantile(calls,[.25,.5,.75],method='linear')
                timing.append({'task':session['task'],'seed':session['seed'],'session':session['session'],'T':T,'engine':e,
                    'median_ms':float(med),'q1_ms':float(q1),'q3_ms':float(q3),'iqr_ms':float(q3-q1),
                    'gpu':session['gpu'],'calls':len(calls)})
    assert len(set(process_ids.values()))==len(manifest['timing_sessions'])
    ti=pd.DataFrame(timing);ti.to_csv(out/'timing_process_medians.csv',index=False)
    for (task,seed,session),g in ti.groupby(['task','seed','session']):
        c={(int(r.T),r.engine):float(r.median_ms) for r in g.itertuples(index=False)}
        timing_pairs.append({'task':task,'seed':int(seed),'session':int(session),
            'eager_T2_saving_percent':100*(1-c[2,'eager']/c[4,'eager']),
            'graph_T2_saving_percent':100*(1-c[2,'graph']/c[4,'graph']),
            'T2_graph_saving_percent':100*(1-c[2,'graph']/c[2,'eager']),
            'T4_graph_saving_percent':100*(1-c[4,'graph']/c[4,'eager']),
            'cross_T2_eager_over_T4_graph':c[2,'eager']/c[4,'graph']})
    pd.DataFrame(timing_pairs).to_csv(out/'timing_paired_ratios.csv',index=False)
    summary=ti.groupby(['task','seed','T','engine']).median_ms.agg(['median','min','max']).reset_index()
    summary.to_csv(out/'timing_checkpoint_summary.csv',index=False)
    report={'status':'passed','scoring_paths':len(rows),'paired_quality_rows':len(pairs),'graph_full_score_comparisons':len(engine),
        'timing_processes':len(manifest['timing_sessions']),'timing_model_units':len(timing)//2,
        'distinct_timing_process_identities':len(set(process_ids.values())),
        'timing_calls':sum(r['calls'] for r in timing),'all_scores_recomputed':True,
        'uncertainty_units':'training seeds are paired; budgets are repeated observations; process and call variation remain nested'}
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(os.environ.get('R2_WORK','/root/snn_revision_round2')))
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();analyze(args.root,args.output)
