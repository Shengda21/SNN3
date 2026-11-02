"""Measured quality-versus-training-time points; never interpolate unknown CE."""
import argparse
from pathlib import Path
import pandas as pd
from common import read, write


def run(root):
    root=Path(root).resolve()
    scores=pd.read_csv(root/'analysis/revision2/quality_scores.csv')
    jobs=read(root/'protocol/C-conf_formal_jobs.json')
    costs={}
    for job in jobs:
        done=read(root/'training'/job['id']/'complete.json')
        for evaluation in done['evaluations']:
            costs[(job['method'],job['T'],job['seed'],evaluation['step'],job['lr'])]=evaluation['training_seconds']
    rows=[]
    for seed in (11021,11022,11023):
        for K in (512,1024,2048):
            native={T:next(j for j in jobs if j['method']=='native' and j['T']==T and j['seed']==seed and 'dual_native_common' in j['roles']) for T in (2,4)}
            mtt=next(j for j in jobs if j['method']=='mtt' and j['seed']==seed)
            dual_seconds=sum(costs[('native',T,seed,K,native[T]['lr'])] for T in (2,4))
            mtt_seconds=costs[('mtt',None,seed,K,mtt['lr'])]
            for e in (2,4):
                for calibration in ('raw','bn_cross'):
                    subset=scores[(scores.cohort=='C-conf') & (scores.seed==seed) & (scores.updates==K)
                        & (scores.infer_T==e) & (scores.calibration==calibration)]
                    if calibration=='bn_cross':subset=subset[subset.cal_T==e]
                    def quality(method,role,T):
                        match=subset[(subset.method==method) & (subset.adapt_T.isna() if T is None else subset.adapt_T==T)]
                        match=match[match.roles.map(lambda text:role in str(text))]
                        assert len(match)==1
                        return match.iloc[0]
                    for strategy,method,role,T,seconds in [('independent_T2_T4','native','dual_native_common',e,dual_seconds),
                        ('MTT_BN_selected','mtt','mtt_joint',None,mtt_seconds)]:
                        q=quality(method,role,T)
                        rows.append({'cohort':'C-conf','seed':seed,'updates':K,'infer_T':e,'calibration':calibration,
                            'strategy':strategy,'training_seconds':seconds,'ce':q.ce,'accuracy':q.accuracy})
    output=root/'analysis/revision2/equal_time_measured_points.csv'
    pd.DataFrame(rows).to_csv(output,index=False)
    bracket=[]
    for seed in (11021,11022,11023):
        group=[r for r in rows if r['seed']==seed]
        target=next(r['training_seconds'] for r in group if r['strategy']=='independent_T2_T4' and r['updates']==2048)
        points=sorted({(r['updates'],r['training_seconds']) for r in group if r['strategy']=='MTT_BN_selected'},key=lambda x:x[1])
        lower=[(K,t) for K,t in points if t<=target]
        upper=[(K,t) for K,t in points if t>=target]
        bracket.append({'seed':seed,'dual_final_training_seconds':target,'MTT_time_lower_measured_point':lower[-1] if lower else None,
            'MTT_time_upper_measured_point':upper[0] if upper else None,'unmeasured_CE_interpolated':False})
    write(root/'analysis/revision2/equal_time_scope.json',{'status':'passed','time_brackets':bracket,
        'scope':'actual saved endpoint quality against measured isolated training-loop device-work time',
        'claim_limit':'time bracketing does not bracket CE or establish equal-time superiority; no artificial 1422-update result',
        'A_historical_costs':'preserved in original archive, not mixed with new-hardware measurements'})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    print(run(parser.parse_args().root))
