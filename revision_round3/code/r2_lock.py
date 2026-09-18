"""Lock real checkpoint identities and complete analysis before test scoring."""
import hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

WORK=Path(os.environ.get('R2_WORK','/root/snn_revision_round2'));ARCHIVE=Path(os.environ.get('R2_ARCHIVE','/root/autodl-tmp/snn0907'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def freeze():
    target=WORK/'results/model_analysis_lock.json'
    if target.exists():return json.loads(target.read_text())
    status=json.loads((WORK/'results/queue_status.json').read_text())
    assert status['phase']=='training_complete',status
    assert (WORK/'results/E0/language_migration.json').exists()
    queue=json.loads((WORK/'protocol/queue.json').read_text())
    selection=json.loads((WORK/'results/selection.json').read_text())['selected']
    gpu_map={8621:0,8622:1,8623:0,9621:0,9622:1,9623:0,9624:1,9625:0}
    init={'language':ARCHIVE/'orthogonal_pilot_20260907/data/fused_state.pt',
          'vision':ARCHIVE/'direction1_budget_rotation_20260907/results/reconstructed_identity_p0.0/best.pt'}
    def checkpoint(task,seed,T,K):
        if task=='language':return ARCHIVE/f'orthogonal_pilot_20260907/data/paper_experiments_20260908/formal/T{T}_seed{seed}_lr2e-05/step{K}.pt'
        return WORK/f'models/formal/s{seed}_T{T}_lr{selection[str(T)]:g}/step{K}.pt'
    rows=[]
    for q in queue['scoring']:
        r=q.copy();task=r['task']
        p=init[task] if r['state']=='shared_initial' else checkpoint(task,r['seed'],r['adapt_T'],r['updates'])
        assert p.is_file(),str(p)
        r.update(checkpoint=None if task=='language' and r['state']=='shared_initial' else str(p),identity_path=str(p),
                 gpu=gpu_map[r['seed']] if 'seed' in r else (0 if task=='language' else 1))
        rows.append(r)
    paths=sorted({r['identity_path'] for r in rows})
    with ThreadPoolExecutor(max_workers=4) as pool:hashes=dict(zip(paths,pool.map(digest,paths)))
    for r in rows:r['checkpoint_sha256']=hashes[r['identity_path']]
    timing=[]
    for q in queue['timing_sessions']:
        r=q.copy();r['gpu']=gpu_map[r['seed']]
        r['checkpoints']={str(T):str(checkpoint(r['task'],r['seed'],T,r['updates'])) for T in [2,4]}
        r['checkpoint_sha256']={str(T):hashes[p] for T,p in [(T,r['checkpoints'][str(T)]) for T in [2,4]]}
        timing.append(r)
    code={p.name:digest(p) for p in (WORK/'code').glob('*.py')}
    assert all(name in code for name in ['r2_score.py','r2_analyze.py','r2_timing.py'])
    sources=json.loads((WORK/'protocol/source_identities.json').read_text())
    source_check={r['path']:digest(ARCHIVE/r['path'])==r['sha256'] for r in sources}
    assert all(source_check.values())
    result={'protocol_version':queue['protocol_version'],'locked_at_unix':time.time(),'scoring':rows,'timing_sessions':timing,
        'selected_vision_lrs':selection,'state_identities':hashes,'execution_and_analysis_code_sha256':code,
        'protocol_sha256':digest(WORK/'protocol/EXPERIMENT_PROTOCOL.md'),'queue_sha256':digest(WORK/'protocol/queue.json'),
        'original_code_identities_match':source_check,'vision_test_scored_before_lock':False,
        'language_test_role':'already inspected test, revision-stage diagnostic; no new model selection',
        'tail_rule':'batch4 for both engines, pad duplicate rows and count original rows only',
        'calibration_seed':9640,'calibration_indices_sha256':digest(WORK/'results/E0/bn_train_ids.npy'),
        'core_contrasts':['P2','P4','I','native_gap','gap_recovery','paired_budget_changes','BN_contrast_changes'],
        'counts':queue['counts']}
    target.write_text(json.dumps(result,indent=2));print(json.dumps({'states':len(hashes),'scoring_paths':len(rows),'timing_sessions':len(timing)}))
    return result


if __name__=='__main__':freeze()
