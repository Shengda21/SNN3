"""CPU integration and software fixtures only. Never creates research results."""
import ast
import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import numpy as np
from common import BUNDLE, protocol, read, require_frozen, sha, verify_prepared, write


def queue_checks(folder):
    from driver import make_formal_jobs,native_selection,plan,scoring_queue
    from cohort_analysis import cohort_contrasts,primary_intervals
    root=Path(folder)
    p=protocol()
    write(root/'environment/protocol.json',p)
    for cohort in ('C-conf','F-arch'):
        directory=root/'construction'/cohort
        directory.mkdir(parents=True)
        cp=directory/'best.pt'
        cp.write_bytes(('SOFTWARE FIXTURE ONLY '+cohort).encode())
        write(directory/'complete.json',{'status':'complete','cohort':cohort,
            'checkpoint':str(cp),'checkpoint_sha256':sha(cp)})
    same={'native_expanded':{'2':2e-5,'4':2e-5},'dual_native_common':{'2':2e-5,'4':2e-5},'mtt_joint':8e-5}
    different={'native_expanded':{'2':1.25e-6,'4':2.5e-6},'dual_native_common':{'2':2e-5,'4':8e-5},'mtt_joint':8e-5}
    assert len(make_formal_jobs(root,'C-conf',same))==9
    C=make_formal_jobs(root,'C-conf',different)
    F=make_formal_jobs(root,'F-arch',same)
    assert len(C)==15 and len(F)==6
    assert sum('native_expanded' in j['roles'] for j in C)==6
    assert sum('dual_native_common' in j['roles'] for j in C)==6
    # Screen selection must let the seven-rate/native and common-five strategy roles differ.
    records={}
    for T in (2,4):
        for rate in p['native_rates']:
            for seed in (11011,11012):
                value=rate*100000
                records[(T,rate,seed)]={'status':'complete','step':2048,'evaluations':[
                    {'step':2048,'raw_validation_by_window':{str(T):{'ce':value,'split':'validation','samples':5000}}}]}
    seven,_=native_selection(records,p['native_rates'],(11011,11012))
    five,_=native_selection(records,p['mtt_rates'],(11011,11012))
    assert set(seven.values())=={1.25e-6} and set(five.values())=={5e-6}
    for cohort,jobs in [('C-conf',C),('F-arch',F)]:
        write(root/'protocol'/f'{cohort}_formal_jobs.json',jobs)
        for job in jobs:
            outputs=[]
            for K in (512,1024,2048):
                cp=root/'models'/job['id']/f'step{K}.pt'
                cp.parent.mkdir(parents=True,exist_ok=True)
                cp.write_bytes(f'SOFTWARE ONLY {job["id"]} {K}'.encode())
                outputs.append({'step':K,'checkpoint':str(cp),'checkpoint_sha256':sha(cp)})
            write(root/'training'/job['id']/'complete.json',{'status':'complete','identity':{'job':job},'checkpoints':outputs})
    with contextlib.redirect_stdout(StringIO()):
        scoring_queue(root)
        scoring_queue(root)
    rows=read(root/'protocol/scoring_queue.json')
    assert len(rows)==352 and len({r['id'] for r in rows})==len(rows)
    assert all(r['updates'] in (0,512,1024,2048) for r in rows)
    assert all(r['calibration']=='raw' or r['cal_seed']==9640 for r in rows)
    quality=[]
    for row in rows:
        value=2.+(row['adapt_T'] or 0)*.03+(row['seed'] or 0)%10*.001
        if row['calibration']!='raw' and row['adapt_T']==2 and row['infer_T']==4:
            value+=row['updates']*.0001
        if 'dual_native_common' in row['roles']:value+=.2
        quality.append({**row,'ce':value})
    Ctests=primary_intervals(cohort_contrasts(quality,'C-conf'),'C-conf')
    Ftests=primary_intervals(cohort_contrasts(quality,'F-arch'),'F-arch')
    assert len(Ctests)==6 and len(Ftests)==3
    assert set(Ctests.n_adaptation_seeds)=={3}
    try:
        require_frozen(root)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Unfrozen protocol was accepted')
    return {'status':'passed','C_formal_dedup_range':[9,15],'F_formal':6,'maximal_CF_score_rows':352,
        'seven_and_five_rate_roles_remain_distinct':True,'C6_F3_integration':True,
        'all_budgets_stop_at_2048':True,'unfrozen_experiments_rejected':True}


def mtt_checks():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from mtt_adapter import official_table,temporal_transfer
    path=BUNDLE/'upstream/temporal_flexibility_in_SNN-main/train-time-stepped/models.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TimeTrans')
    namespace={'nn':nn,'torch':torch,'utils':SimpleNamespace(table_spwn=official_table)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    reference=namespace['TimeTrans'](0)
    for incoming in (2,3,4):
        for outgoing in (2,3,4):
            x=torch.randn(incoming,2,3,4,requires_grad=True)
            y=x.detach().transpose(0,1).contiguous().requires_grad_()
            actual=temporal_transfer(x,outgoing)
            expected=reference(y,outgoing).transpose(0,1)
            assert torch.equal(actual,expected)
            weights=torch.randn_like(actual)
            (actual*weights).sum().backward();(expected*weights).sum().backward()
            assert torch.equal(x.grad,y.grad.transpose(0,1))
    return {'status':'passed','official_TimeTrans_forward_and_gradient_pairs':9,'GPU_work':False}


def main():
    from prepare import prepare
    from e3 import self_check as E3_check
    from cohort_analysis import self_check as analysis_check
    from normalization import self_check as BN_check
    from runtime import sample_plan,split
    files=[x for x in BUNDLE.rglob('*.py') if '__pycache__' not in x.parts]
    for file in files:ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
    official_C=read(BUNDLE/'vendor/spikingresformer_reference/PROVENANCE.json')
    for name,expected in official_C['official_git_blob_sha1'].items():
        data=(BUNDLE/'vendor/spikingresformer_reference'/name).read_bytes()
        import hashlib
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==expected
    data=BUNDLE.parent/'cloud_revision2_data_20261020'
    for row in read(BUNDLE/'data_manifest.json'):
        source=data/row['path']
        assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
    import hashlib
    with (data/'cifar-100-python.tar.gz').open('rb') as stream:
        assert hashlib.file_digest(stream,'md5').hexdigest()=='eb9058c3a382ffc7106e4002c42a8d85'
    assets=BUNDLE.parent/'cloud_revision2_A_assets_20261020'
    manifest=read(BUNDLE/'a_asset_manifest.json')
    for row in manifest['screen_sources']+[manifest['retained_start']]:
        source=assets/row['path']
        assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
    with tempfile.TemporaryDirectory(prefix='software_check_',dir=BUNDLE) as temporary:
        parent=Path(temporary).resolve()
        assert parent.is_relative_to(BUNDLE)
        root=parent/'prepared'
        with contextlib.redirect_stdout(StringIO()):
            prepare(root,data)
            prepare(root,data)
        verify_prepared(root)
        assert not list(root.rglob('*.pt')) and not list(root.rglob('complete.json'))
        small,small_aug=sample_plan(root,11021,512)
        large,large_aug=sample_plan(root,11021,2048)
        assert np.array_equal(small,large[:len(small)]) and np.array_equal(small_aug,large_aug[:len(small)])
        division=split(root)
        assert set(large).issubset(set(division['train'])) and not set(large)&set(division['validation'])
        calibration=np.random.default_rng(9640).choice(np.asarray(division['train']),2048,replace=False)
        assert len(set(calibration))==2048 and not set(calibration)&set(division['validation'])
        queue_report=queue_checks(parent/'software_queue')
    backbone_report=None
    try:
        from backbone import self_check as backbone_check
        backbone_report=backbone_check()
    except ImportError as exc:
        backbone_report={'status':'dependency_unavailable','reason':str(exc),'full_model_validation_claimed':False}
    report={'status':'passed','scope':'CPU software checks only; no GPU research experiment',
        'python_files_parsed':len(files),'data_assets_hashed':2,'A_existing_weight_assets_hashed':11,
        'CIFAR_official_MD5':'passed','clean_prepare_idempotence':'passed',
        'C_official_reference_yaml_main_git_blob_SHA1':'passed',
        'prepare_creates_no_model_or_research_completion_records':True,'paired_sample_prefix':'passed',
        'calibration_train_only':'passed','queue_integration':queue_report,
        'E3':E3_check(),'cohort_analysis':analysis_check(),'BN':BN_check(),'MTT':mtt_checks(),
        'F_backbone':backbone_report,'GPU_experiments_executed':False,
        'local_git_freeze':'not_performed; formal GPU driver remains blocked until verified freeze',
        'GPU_pending':['CUDA/CuPy kernels','full C/F forward-backward and resume','C physical batch128 AMP',
                       'MTT on full C backbone','graph/eager ABA and fresh-process timing','clean checkpoint replay'],
        'protocol_sha256':sha(BUNDLE/'protocol.json')}
    write(BUNDLE/'software_validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
