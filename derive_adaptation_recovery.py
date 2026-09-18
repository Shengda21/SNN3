"""Recompute the original four-budget recovery description from audited scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def main():
    q=pd.read_csv(ROOT/'recomputed/quality_recomputed.csv')
    q=q[q.split=='test']
    zero={T:float(q[(q.stage=='frozen')&(q['T']==T)].iloc[0].ce) for T in [2,4]}
    d0=zero[2]-zero[4];rows=[]
    seeds=[8621,8622,8623]
    for K in [512,1024,2048,4096]:
        gaps=[]
        for seed in seeds:
            g=q[(q.stage=='formal')&(q.seed==seed)&(q.step==K)].set_index('T')
            gaps.append(float(g.loc[2,'ce']-g.loc[4,'ce']))
        rows.append({'updates':K,'gap_each_seed':gaps,'gap_mean':float(np.mean(gaps)),
            'gap_recovery_each_seed':[d0-g for g in gaps],
            'gap_recovery_mean':d0-float(np.mean(gaps)),
            'initial_gap_recovered_percent':100*(d0-float(np.mean(gaps)))/d0})
    a,b=rows[0],rows[-1]
    increases=[]
    for earlier,later in zip(rows,rows[1:]):
        for index,seed in enumerate(seeds):
            change=later['gap_each_seed'][index]-earlier['gap_each_seed'][index]
            if change>0:increases.append({'seed':seed,'from_updates':earlier['updates'],'to_updates':later['updates'],'gap_increase':change})
    result={'source':'original 2026-09-08 full-test records, natural batch-two tail','seeds':seeds,
        'initial_gap':d0,'budgets':rows,
        'early_fraction_of_final_mean_recovery_percent':100*a['gap_recovery_mean']/b['gap_recovery_mean'],
        'early_fraction_each_seed_percent':[100*x/y for x,y in zip(a['gap_recovery_each_seed'],b['gap_recovery_each_seed'])],
        'local_gap_increases':increases,'statistic':'ratio of mean recoveries; within-seed ratios kept separately'}
    (ROOT/'recomputed/adaptation_recovery.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
