"""Revision-2 driver. Default plan is CPU-only; every GPU stage is explicit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import (BUNDLE, environment, exclusive, protocol, read, require_frozen,
                    sha, verify_file, verify_prepared, write)


def save_locked(path,value):
    if Path(path).exists():
        assert read(path)==value, f'Locked manifest changed: {path}'
    else:
        write(path,value)


def plan():
    p=protocol()
    result={'status':'prepared_plan_only_no_GPU','construction_jobs':2,'construction_epochs_total':630,
        'C_native_screen':28,'C_MTT_screen':10,'F_native_screen':28,
        'C_formal_min':9,'C_formal_max':15,'F_formal':6,'A_E3_formal_if_rate_changes':3,
        'updates_per_trajectory':2048,'budgets':p['budgets'],
        'C_primary_family_size':6,'F_primary_family_size':3,
        'new_tests_unit':'adaptation seed conditional on each retained start',
        'protocol_status':p['status'],
        'default_starts_GPU_training':False,'full_old_three_by_600_or_4096_template_used':False}
    print(json.dumps(result,indent=2))
    return result


def command(root,name,script,*args):
    if shutil.disk_usage(root).free/2**30<10:
        raise RuntimeError('Less than 10 GiB free; preserve results before continuing')
    cmd=[sys.executable,str(BUNDLE/'scripts'/script),'--root',str(root),*map(str,args)]
    started=time.time()
    log=root/'logs'/f'{name}.log'
    with log.open('a',encoding='utf-8') as stream:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    write(root/'state'/f'{name}.json',{'status':'complete' if result.returncode==0 else 'failed',
        'command':cmd,'returncode':result.returncode,'started':started,'process_seconds':time.time()-started,
        'time_scope':'subprocess interval including setup, validation and IO; not isolated training or energy'})
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {log}; seeds and failures are not silently replaced')


def start(root,cohort):
    path=root/'construction'/cohort/'complete.json'
    result=read(path)
    assert result['status']=='complete' and result['cohort']==cohort
    verify_file(result['checkpoint'],result['checkpoint_sha256'])
    return result['checkpoint'],result['checkpoint_sha256']


def make_job(cohort,stage,method,T,seed,lr,initial,initial_sha,roles):
    architecture=protocol()['cohorts'][cohort]['architecture'] if cohort in protocol()['cohorts'] else 'resformer'
    return {'id':f'{cohort}__{stage}__{method}_s{seed}_T{T}_lr{lr:g}',
        'cohort':cohort,'stage':stage,'architecture':architecture,'method':method,'T':T,
        'seed':seed,'lr':lr,'steps':2048,'budgets':[512,1024,2048],
        'initial_checkpoint':str(initial),'initial_checkpoint_sha256':initial_sha,'roles':sorted(roles)}


def screen_jobs(root,cohort):
    p=protocol();c=p['cohorts'][cohort]
    initial,fingerprint=start(root,cohort)
    jobs=[]
    for seed in c['screen_seeds']:
        for rate in p['native_rates']:
            for T in (2,4):
                jobs.append(make_job(cohort,'screen','native',T,seed,rate,initial,fingerprint,['native_screen']))
        if cohort=='C-conf':
            for rate in p['mtt_rates']:
                jobs.append(make_job(cohort,'screen','mtt',None,seed,rate,initial,fingerprint,['mtt_screen']))
    return jobs


def execute_jobs(root,jobs):
    for job in jobs:
        path=root/'protocol/jobs'/f'{job["id"]}.json'
        save_locked(path,job)
        command(root,job['id'],'train_job.py','--job',path)


def native_selection(records,rates,seeds):
    rows=[]
    for T in (2,4):
        for rate in rates:
            values=[]
            for seed in seeds:
                result=records[(T,rate,seed)]
                assert result['status']=='complete' and result['step']==2048
                evaluation=next(x for x in result['evaluations'] if x['step']==2048)
                value=evaluation['raw_validation_by_window'][str(T)]
                assert value['split']=='validation' and value['samples']==5000
                values.append(value['ce'])
            rows.append({'T':T,'lr':rate,'ce_each':values,'mean_ce':sum(values)/len(values)})
    selected={str(T):min((r for r in rows if r['T']==T),key=lambda r:(r['mean_ce'],r['lr']))['lr'] for T in (2,4)}
    return selected,rows


def make_formal_jobs(root,cohort,selection):
    c=protocol()['cohorts'][cohort]
    initial,fingerprint=start(root,cohort)
    jobs={}
    def add(method,T,seed,lr,role):
        key=(method,T,seed,lr)
        if key not in jobs:
            jobs[key]=make_job(cohort,'formal',method,T,seed,lr,initial,fingerprint,[])
        jobs[key]['roles']=sorted(set(jobs[key]['roles']+[role]))
    for seed in c['formal_seeds']:
        for T in (2,4):
            add('native',T,seed,selection['native_expanded'][str(T)],'native_expanded')
            if cohort=='C-conf':
                add('native',T,seed,selection['dual_native_common'][str(T)],'dual_native_common')
        if cohort=='C-conf':
            add('mtt',None,seed,selection['mtt_joint'],'mtt_joint')
    return list(jobs.values())


def select(root,cohort):
    p=protocol();c=p['cohorts'][cohort]
    jobs=read(root/'protocol'/f'{cohort}_screen_jobs.json')
    native={}
    mtt=[]
    for job in jobs:
        done=read(root/'training'/job['id']/'complete.json')
        assert done['status']=='complete' and done['identity']['job']==job
        for source in done['checkpoints']:
            verify_file(source['checkpoint'],source['checkpoint_sha256'])
        if job['method']=='native':
            native[(job['T'],job['lr'],job['seed'])]=done
        else:
            mtt.append((job,done))
    expanded,all_rows=native_selection(native,p['native_rates'],c['screen_seeds'])
    selection={'cohort':cohort,'native_expanded':expanded,'native_seven_rate_candidates':all_rows,
               'native_criterion':'raw native validation CE','test_used':False,
               'protocol_sha256':sha(BUNDLE/'protocol.json')}
    if cohort=='C-conf':
        common,common_rows=native_selection(native,p['mtt_rates'],c['screen_seeds'])
        selection['dual_native_common']=common
        selection['dual_common_five_rate_candidates']=common_rows
        values=[]
        for job,done in mtt:
            source=next(row for row in done['checkpoints'] if row['step']==2048)
            verify_file(source['checkpoint'],source['checkpoint_sha256'])
            cached=root/'selection_eval'/f'{job["id"]}_matched_BN.json'
            command(root,'select_BN_'+job['id'],'evaluate_checkpoint.py',
                    '--architecture','resformer','--checkpoint',source['checkpoint'],'--output',cached)
            result=read(cached)
            assert result['identity']['checkpoint_sha256']==source['checkpoint_sha256'], 'Calibrated selection used a changed screen source'
            values.append({'seed':job['seed'],'lr':job['lr'],
                           'ce':result['joint_matched_BN_validation_ce'],
                           'checkpoint_sha256':source['checkpoint_sha256']})
        candidates=[]
        for rate in p['mtt_rates']:
            group=[x for x in values if x['lr']==rate]
            assert len(group)==2 and {x['seed'] for x in group}==set(c['screen_seeds'])
            candidates.append({'lr':rate,'mean_ce':sum(x['ce'] for x in group)/2,'seeds':group})
        selection['mtt_joint']=min(candidates,key=lambda x:(x['mean_ce'],x['lr']))['lr']
        selection['mtt_candidates']=candidates
        selection['MTT_criterion']='matched-BN validation CE, two-window then two-seed mean'
        selection['policy_difference_from_original_A']='C MTT uses calibrated validation selection; original A raw-selected family remains historical and separate'
    save_locked(root/'protocol'/f'{cohort}_selection.json',selection)
    formal=make_formal_jobs(root,cohort,selection)
    save_locked(root/'protocol'/f'{cohort}_formal_jobs.json',formal)
    print(f'{cohort}: selected on validation; {len(formal)} distinct formal trajectories')


def formal_A(root):
    selection=read(root/'results/e3/A-conf-BN-selection/selection.json')
    rows=read(root/'results/e3/A-conf-BN-selection/conditional_formal_tasks.json')
    if not selection['conditional_retrain_needed']:
        assert not rows
        print('A calibrated criterion retains old rate; no A training launched')
        return
    jobs=[]
    for row in rows:
        jobs.append({**row,'architecture':'resformer','T':None,'stage':'formal',
                     'roles':['mtt_joint'],'policy_id':'A-conf-BN-selection'})
    save_locked(root/'protocol/A-conf-BN-selection_formal_jobs.json',jobs)
    execute_jobs(root,jobs)


def scoring_queue(root):
    p=protocol();rows=[];sources=[]
    def add(cohort,architecture,method,seed,T,K,path,fingerprint,roles):
        verify_file(path,fingerprint)
        source_id=f'{cohort}_{method}_s{seed}_T{T}_K{K}_{fingerprint[:12]}'
        base={'cohort':cohort,'architecture':architecture,'method':method,'seed':seed,
              'adapt_T':T,'updates':K,'checkpoint':str(path),'checkpoint_sha256':fingerprint,
              'roles':roles,'source_id':source_id}
        sources.append(base)
        for e in (2,4):
            rows.append({**base,'id':f'{source_id}_raw_e{e}','infer_T':e,'calibration':'raw'})
        for cal in (2,4):
            for e in ((2,4) if cohort!='F-arch' else (cal,)):
                rows.append({**base,'id':f'{source_id}_c{cal}_e{e}_BN9640','infer_T':e,
                    'calibration':'bn_cross','cal_T':cal,'cal_seed':9640})
    for cohort,c in p['cohorts'].items():
        initial,fingerprint=start(root,cohort)
        add(cohort,c['architecture'],'initial',None,None,0,initial,fingerprint,['retain'])
        for job in read(root/'protocol'/f'{cohort}_formal_jobs.json'):
            done=read(root/'training'/job['id']/'complete.json')
            assert done['status']=='complete' and done['identity']['job']==job
            for cp in done['checkpoints']:
                add(cohort,c['architecture'],job['method'],job['seed'],job['T'],cp['step'],
                    cp['checkpoint'],cp['checkpoint_sha256'],job['roles'])
    extra=root/'protocol/A-conf-BN-selection_formal_jobs.json'
    if extra.exists():
        for job in read(extra):
            done=read(root/'training'/job['id']/'complete.json')
            assert done['status']=='complete' and done['identity']['job']==job
            for cp in done['checkpoints']:
                add(job['cohort'],'resformer','mtt',job['seed'],None,cp['step'],cp['checkpoint'],cp['checkpoint_sha256'],job['roles'])
    path=root/'protocol/scoring_queue.json'
    save_locked(path,rows)
    save_locked(root/'protocol/source_lock.json',{'queue_sha256':sha(path),
        'protocol_sha256':sha(BUNDLE/'protocol.json'),'sources':sources,
        'execution_protocol_sha256':sha(root/'environment/protocol.json'),
        'locked_before_new_test_scoring':True,'test_history':'initial-submission test results already known'})
    print(f'Locked {len(rows)} scoring rows; no test evaluation launched by this action')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='plan',choices=['plan','prepare','import-A','preflight','smoke',
        'freeze-protocol','construct','screen','select','formal','E3','formal-A','lock-score','score','timing','analyze'])
    parser.add_argument('--root',type=Path,default=BUNDLE/'runs/revision2')
    parser.add_argument('--cohort',choices=['C-conf','F-arch'])
    parser.add_argument('--data',type=Path,default=BUNDLE.parent/'cloud_revision2_data_20261020')
    parser.add_argument('--A-assets',dest='A_assets',type=Path,default=BUNDLE.parent/'cloud_revision2_A_assets_20261020')
    a=parser.parse_args()
    if a.action=='plan':
        plan();return
    if a.action=='prepare':
        from prepare import prepare
        prepare(a.root,a.data);return
    root=verify_prepared(a.root)
    environment(root)
    if a.action=='import-A':
        from prepare import import_A
        import_A(root,a.A_assets);return
    if a.action=='freeze-protocol':
        from freeze_protocol import freeze
        freeze(root);return
    if a.action in ('construct','screen','select','formal') and not a.cohort:
        parser.error('--cohort is required for this explicit GPU stage')
    with exclusive(root):
        if a.action in ('preflight','smoke'):
            command(root,a.action,a.action+'.py')
        else:
            require_frozen(root)
            if a.action not in ('lock-score','analyze'):
                assert read(root/'environment/model_smoke.json')['status']=='passed'
            if a.action=='construct':
                command(root,'construct_'+a.cohort,'train_job.py','--construct',a.cohort)
            elif a.action=='screen':
                jobs=screen_jobs(root,a.cohort)
                save_locked(root/'protocol'/f'{a.cohort}_screen_jobs.json',jobs)
                execute_jobs(root,jobs)
            elif a.action=='select':
                select(root,a.cohort)
            elif a.action=='formal':
                execute_jobs(root,read(root/'protocol'/f'{a.cohort}_formal_jobs.json'))
            elif a.action=='E3':
                from e3 import evaluate_sources
                evaluate_sources(root)
            elif a.action=='formal-A':
                formal_A(root)
            elif a.action=='lock-score':
                scoring_queue(root)
            elif a.action=='score':
                command(root,'score_all','score.py')
            elif a.action=='timing':
                from measure import run
                run(root,command)
            elif a.action=='analyze':
                command(root,'cohort_analysis','cohort_analysis.py')
                command(root,'quality_cost_view','cost_view.py')


if __name__=='__main__':
    main()
