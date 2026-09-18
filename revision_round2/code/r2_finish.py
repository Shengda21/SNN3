"""Complete the frozen quality/timing queue after training, without auto-shutdown."""
import concurrent.futures,json,os,subprocess,sys,time,traceback
from pathlib import Path
WORK=Path(os.environ.get('R2_WORK','/root/snn_revision_round2'));PY=sys.executable

def write(name,obj):
    path=WORK/'results'/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def command(name,args,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (WORK/'logs'/f'{name}.log').open('a') as f:
        p=subprocess.run([PY]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{name} failed with exit {p.returncode}; see logs/{name}.log')

def main():
    while True:
        status=json.loads((WORK/'results/queue_status.json').read_text())
        if status['phase']=='training_complete':break
        if status['phase'] not in ['screening','formal']:raise RuntimeError(str(status))
        time.sleep(20)
    if not (WORK/'results/E0/language_migration.json').exists():
        command('language_checks',[str(WORK/'code/r2_score.py'),'check'],0)
    command('model_analysis_lock',[str(WORK/'code/r2_lock.py')])
    manifest=json.loads((WORK/'results/model_analysis_lock.json').read_text())
    # Reject a silent code edit after the scoring/analysis lock.
    import hashlib
    for name,expected in manifest['execution_and_analysis_code_sha256'].items():
        assert hashlib.sha256((WORK/'code'/name).read_bytes()).hexdigest()==expected,name
    write('finish_status.json',{'phase':'full_scoring','time':time.time(),'paths':len(manifest['scoring'])})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(command,f'scoring_gpu{gpu}',[str(WORK/'code/r2_score.py'),'worker','--gpu',str(gpu)],gpu) for gpu in [0,1]]
        for f in futures:f.result()
    for row in manifest['scoring']:assert (WORK/'results/scoring'/row['id']/'complete.json').exists(),row['id']
    write('scoring_finished.json',{'paths':len(manifest['scoring']),'time':time.time()})
    if (WORK/'results/backup_active.json').exists():
        expected=json.loads((WORK/'results/backup_active.json').read_text())['session']
        write('finish_status.json',{'phase':'waiting_for_backup_pause','time':time.time()})
        while True:
            paused=WORK/'results/backup_paused.json'
            if paused.exists() and json.loads(paused.read_text()).get('session')==expected:break
            time.sleep(2)
    write('finish_status.json',{'phase':'isolated_timing','time':time.time()})
    # All scoring children have exited. Timing sessions run one at a time across the instance.
    for session in manifest['timing_sessions']:
        command('timing_'+session['id'],[str(WORK/'code/r2_timing.py'),'--id',session['id']],session['gpu'])
    command('analysis',[str(WORK/'code/r2_analyze.py')])
    write('finish_status.json',{'phase':'experiments_complete_pending_download','time':time.time()})
    print('EXPERIMENTS_COMPLETE_PENDING_DOWNLOAD',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write('finish_status.json',{'phase':'failed','error':traceback.format_exc(),'time':time.time()})
        raise
