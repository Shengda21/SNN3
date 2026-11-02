import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[1]))
"""Original noninferiority and endpoint equations on the new complete paired matrix."""
import argparse,itertools,statistics,math
from port_common import *

def assess(rows,units):
    index={}
    for r in rows:
        require(r['unit'] in units and r['method'] in ('S0','reuse2','conditional'),'declared paired unit/method')
        require(r['split']=='validation' and r['samples']==4050,'full validation gate')
        require(math.isfinite(r['ordinary_CE']) and r['ordinary_CE']>=0 and math.isfinite(r['accuracy']) and 0<=r['accuracy']<=1,'finite real metrics')
        require(abs(round(4050*r['accuracy'])-4050*r['accuracy'])<=1e-8,'accuracy from complete4050 example counts')
        key=(r['unit'],r['method'],r['K'],r['window'],r['BN']);require(key not in index,'duplicate route');index[key]=r
    expected={(u,m,k,t,b) for u,m,k,t,b in itertools.product(units,('reuse2','conditional'),BUDGETS,(2,4),('raw','matched'))}
    expected|={(u,'S0',0,t,b) for u,t,b in itertools.product(units,(2,4),('raw','matched'))}
    require(set(index)==expected,'complete unique paired full-budget matrix')
    require(len({r['example_identity_sha256'] for r in rows})==1 and len({r['label_identity_sha256'] for r in rows})==1,'identical paired validation examples and labels')
    for u in units:require(len({r['start_sha256'] for r in rows if r['unit']==u})==1,'same actual S0 within paired unit')
    conditions=[];lookup={}
    for k,t,b,baseline in itertools.product(BUDGETS,(2,4),('raw','matched'),('S0','reuse2')):
        gains=[];accuracies=[]
        for u in units:
            c=index[u,'conditional',k,t,b];r=index[u,baseline,0 if baseline=='S0' else k,t,b]
            gains.append(r['ordinary_CE']-c['ordinary_CE'])
            accuracies.append((round(4050*c['accuracy'])-round(4050*r['accuracy']))/4050)
        row={'K':k,'window':t,'BN':b,'comparator':baseline,'CE_gain':gains,'accuracy_gain':accuracies,
             'mean_CE_gain':statistics.mean(gains),'mean_accuracy_gain':statistics.mean(accuracies)}
        conditions.append(row);lookup[k,t,b,baseline]=row
    endpoints=[]
    for b in ('raw','matched'):
        checks={'T4_accuracy_strictly_above_reuse':lookup[2048,4,b,'reuse2']['mean_accuracy_gain']>0,
                'T4_accuracy_retains_S0':lookup[2048,4,b,'S0']['mean_accuracy_gain']>=0,
                'T2_accuracy_retains_reuse':lookup[2048,2,b,'reuse2']['mean_accuracy_gain']>=0,
                'T2_accuracy_retains_S0':lookup[2048,2,b,'S0']['mean_accuracy_gain']>=0,
                'joint_CE_strictly_better_than_reuse':statistics.mean(lookup[2048,t,b,'reuse2']['mean_CE_gain'] for t in (2,4))>0}
        endpoints.append({'BN':b,'checks':checks,'passed':all(checks.values())})
    accuracy=all(r['mean_accuracy_gain']>=0 for r in conditions);ce=all(r['mean_CE_gain']>=0 for r in conditions)
    endpoint=all(r['passed'] for r in endpoints)
    return {'status':'complete','units':units,'endpoint_gate_passed':endpoint,'full_budget_accuracy_target_met':accuracy,
            'full_budget_CE_target_met':ce,'gate_passed':endpoint and accuracy and ce,'conditions':conditions,
            'endpoints':endpoints,'test_used':False,'statistical_significance_established':False}

def bound_assess(rows_path,units,stage,construction_root):
    verify_sources();rows=read(rows_path);construction_root=Path(construction_root).resolve()
    expected_units=['dev34611','dev34612'] if stage=='development' else ['formal34621','formal34622','formal34623']
    require(units==expected_units,'frozen paired-unit names')
    sources=[]
    for row in rows:
        path=Path(row['score_source']).resolve();require(sha(path)==row['score_source_sha256'],'bound actual score receipt')
        actual=read(path);require(not actual.get('software_fixture',False),'software scores cannot authorize scientific gates')
        for key in ('split','samples','window','BN','ordinary_CE','accuracy','example_identity_sha256','label_identity_sha256','start_sha256'):
            require(row[key]==actual[key],'gate score value mismatch '+key)
        require(actual['schema']==SCHEMA and actual['authority']['source_freeze_sha256']==source_hash(),'score method freeze')
        role=actual['authority']['role']
        if stage=='development':
            require(actual['authority']['construction_root']==str(construction_root) and role=='development','actual paired development root')
        else:require(Path(actual['authority']['construction_root']).parent==construction_root and role in ('formal1','formal2','formal3'),'actual formal construction roots')
        require(actual['checkpoint_step']==row['K'],'actual checkpoint budget')
        seed=int(row['unit'].removeprefix('dev').removeprefix('formal'))
        if row['method']!='S0':require(actual['adaptation_seed']==seed,'actual paired adaptation seed')
        if stage=='formal':require(role=={34621:'formal1',34622:'formal2',34623:'formal3'}[seed],'formal actual role-seed')
        require((row['method']=='S0' and actual['method']=='S0') or
                (row['method']=='reuse2' and actual['method']=='native' and actual['train_T']==2) or
                (row['method']=='conditional' and actual['method']=='conditional'),'actual method comparator')
        sources.append({'path':str(path),'sha256':sha(path)})
    result=assess(rows,units)
    result.update(schema=SCHEMA,stage=stage,construction_root=str(construction_root),
                  paired_seeds=[34611,34612] if stage=='development' else [34621,34622,34623],
                  source_freeze_sha256=source_hash(),score_sources=sources,
                  rows_path=str(Path(rows_path).resolve()),rows_sha256=sha(rows_path))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',required=True);p.add_argument('--units',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--stage',choices=['development','formal'],required=True);p.add_argument('--construction-root',required=True)
    args=p.parse_args();write(args.out,bound_assess(args.rows,args.units,args.stage,args.construction_root))
