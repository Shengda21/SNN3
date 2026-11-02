"""Audit all saved logits and summarize three independent starts without pseudoreplication."""
import csv,itertools,math,statistics
from pathlib import Path
import numpy as np
from port_common import *

def analyze(receipts_path,out,outcome='scientific_complete'):
    require(outcome in ('scientific_complete','development_negative','formal_negative'),'explicit analysis outcome')
    receipts=read(receipts_path);rows=[];all_rows=[]
    for record in receipts:
        path=Path(record['path']);require(sha(path)==record['sha256'] and read(path)==record['row'],'immutable actual score receipt')
        row=record['row'];arrays=path.parent/'scores.npz';require(sha(arrays)==row['arrays_sha256'],'immutable score arrays')
        with np.load(arrays,allow_pickle=False) as a:
            z,y,ids=(a[k] for k in ('logits','labels','example_id'))
            require(z.shape==(row['samples'],10) and z.dtype==np.float32 and np.isfinite(z).all(),'all actual native logits')
            require(hashlib.sha256(ids.tobytes()).hexdigest()==row['example_identity_sha256'] and
                    hashlib.sha256(y.tobytes()).hexdigest()==row['label_identity_sha256'],'actual paired identities')
            x=z.astype(np.float64);x-=x.max(1,keepdims=True);loss=np.log(np.exp(x).sum(1))-x[np.arange(len(y)),y]
            require(float(loss.mean())==row['ordinary_CE'] and float(np.mean(z.argmax(1)==y))==row['accuracy'],'independent exact metric recomputation')
        all_rows.append(row)
        if row['authority']['role'].startswith('formal'):rows.append(row)
    require(bool(all_rows),'no analysis of an empty experiment')
    if outcome!='scientific_complete':require(all(r['split']=='validation' for r in all_rows),'negative gates cannot authorize test reads')
    if outcome=='development_negative':require(not rows,'no formal claims from development-only evidence')
    groups={}
    for r in rows:
        key=(r['split'],r['variant'],r['checkpoint_step'],r['window'],r['BN']);groups.setdefault(key,[]).append(r)
    summaries=[]
    for key,selected in sorted(groups.items()):
        require(len(selected)==3 and {r['authority']['role'] for r in selected}=={'formal1','formal2','formal3'},'three independently constructed starts')
        selected=sorted(selected,key=lambda r:r['authority']['role'])
        summaries.append({'split':key[0],'variant':key[1],'K':key[2],'window':key[3],'BN':key[4],
            'mean_CE':statistics.mean(r['ordinary_CE'] for r in selected),'sample_SD_CE':statistics.stdev(r['ordinary_CE'] for r in selected),
            'mean_accuracy':statistics.mean(r['accuracy'] for r in selected),'sample_SD_accuracy':statistics.stdev(r['accuracy'] for r in selected),
            'individual_CE':[r['ordinary_CE'] for r in selected],'individual_accuracy':[r['accuracy'] for r in selected]})
    lookup={(r['split'],r['authority']['role'],r['variant'],r['checkpoint_step'],r['window'],r['BN']):r for r in rows};paired=[]
    for split,k,b in itertools.product(sorted({r['split'] for r in rows}),BUDGETS,('raw','matched')):
        values=[]
        for role in ('formal1','formal2','formal3'):
            candidate=statistics.mean(lookup[split,role,'conditional',k,t,b]['ordinary_CE'] for t in (2,4))
            reference=statistics.mean((lookup[split,role,'native2',k,2,b]['ordinary_CE'],lookup[split,role,'native4',k,4,b]['ordinary_CE']))
            values.append(candidate-reference)
        paired.append({'split':split,'K':k,'BN':b,'candidate_minus_dedicated_dual_equal_window_CE':values,
                       'mean':statistics.mean(values),'sample_SD':statistics.stdev(values)})
    result={'status':'complete','schema':SCHEMA,'outcome':outcome,
            'independent_formal_starts':len({r['authority']['role'] for r in rows}),
            'development_evidence_scope':'one construction start, two paired adaptation streams; screening rates are not independent starts',
            'score_receipts_sha256':sha(receipts_path),'all_logits_metrics_independently_recomputed':True,
            'summaries':summaries,'paired_primary_descriptions':paired,'statistical_significance_claim':False,
            'test_scored':any(r['split']=='test' for r in all_rows),'verified_routes':len(all_rows),
            'test_used_for_selection':False,'source_freeze_sha256':sha(HERE/'SOURCE_FREEZE.json')}
    out=Path(out);write(out/'summary.json',result)
    write(out/'individual_scores.json',all_rows)
    if summaries:
        with (out/'summary.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    return result
