"""Isolated latency -> locked confirmation -> analysis -> verified-download handoff."""
import concurrent.futures,hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/paper_experiments_20260908';BASE=ROOT.parent

def save(name,data):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def call(script,args,log,gpu=0):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (OUT/'logs'/log).open('a') as f:r=subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{script} {args} failed with exit {r.returncode}, log {log}')

def benchmark():
    marker=OUT/'latency_phase_active';marker.touch()
    save('execution_status.json',{'stage':'isolated_latency','status':'running','time':time.time()})
    # Local backup acknowledges pause between file transfers before timings begin.
    deadline=time.time()+600
    while not (OUT/'backup_paused').exists():
        if time.time()>deadline:raise RuntimeError('Backup did not acknowledge latency pause within 10 minutes')
        time.sleep(3)
    try:
        for session in [8801,8802,8803]:
            call('paper_bench.py',['--group','historical','--session',session,'--copy-control'],f'latency_language_{session}.log')
        status=json.loads((OUT/'vision_status.json').read_text()) if (OUT/'vision_status.json').exists() else {'state':'missing'}
        if status.get('state') not in ['ready_original','ready_reconstructed']:raise RuntimeError('Visual branch not ready; restore or repair reconstruction before finalization')
        for session in [8801,8802,8803]:
            call('paper_bench.py',['--architecture','vision','--checkpoint',status['checkpoint'],'--tag',status['state'],'--group','historical','--session',session,'--copy-control'],f'latency_vision_{session}.log')
        for stage,step in [('formal',4096),('legacy',512)]:
            for p in sorted((OUT/stage).glob('*/complete.json')):
                r=json.loads(p.read_text());c=r['config'];tag=f'{"formal" if stage=="formal" else "legacy_reconstructed"}_T{c["T"]}_seed{c["seed"]}_step{step}'
                args=['--group',stage,'--checkpoint',r['checkpoint'],'--T',c['T'],'--session',c['seed'],'--tag',tag]
                if stage=='legacy':args+=['--batches',1,4]
                call('paper_bench.py',args,f'latency_{tag}.log',gpu=int(c['visible_devices']))
        call('paper_bench.py',['--group','masked','--masked'], 'latency_masked.log')
        call('paper_bench.py',['--masked-quality'],'masked_quality.log')
        save('benchmark_finished.json',{'time':time.time(),'both_architectures':True})
    finally:marker.unlink(missing_ok=True)

def archive_manifest():
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    folders=[ROOT/'code',ROOT/'vendor',ROOT/'protocol',ROOT/'environment',ROOT/'data',OUT,
        BASE/'direction1_budget_rotation_20260907/code',BASE/'direction1_budget_rotation_20260907/vendor',
        BASE/'direction1_budget_rotation_20260907/data',BASE/'direction1_budget_rotation_20260907/results/reconstructed_identity_p0.0']
    files=[];excluded={'archive_manifest.json','archive_ready.json','backup_paused','latency_phase_active','source_bundle.tar.gz'}
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or p.is_symlink() or any(s in p.parts for s in ['.git','__pycache__']) or p.suffix in ['.tmp','.partial'] or p.name in excluded:continue
            files.append({'path':p.relative_to(BASE).as_posix(),'size':p.stat().st_size,'sha256':sha(p)})
    save('archive_manifest.json',{'files':files,'file_count':len(files),'total_bytes':sum(f['size'] for f in files),'created':time.time()})
    save('archive_ready.json',{'all_experiment_tasks_finished':True,'download_verification_required_before_shutdown':True,'file_count':len(files),'time':time.time()})

def main():
    assert (OUT/'training_finished.json').exists()
    finished=json.loads((OUT/'training_finished.json').read_text())
    assert finished.get('formal_complete') and finished.get('legacy_reconstruction_complete'),'Training queue contains unfinished runs'
    for stage,expected in [('screen',16),('formal',6),('legacy',6)]:
        assert len(list((OUT/stage).glob('*/complete.json')))==expected,f'{stage}: frozen run count incomplete'
    if not (OUT/'benchmark_finished.json').exists():benchmark()
    call('paper_confirm.py',['freeze'],'confirmation_freeze.log')
    call('paper_confirm.py',['prepare'],'confirmation_data.log')
    save('execution_status.json',{'stage':'locked_confirmation','status':'running','time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(call,'paper_confirm.py',['worker','--gpu',gpu],f'confirmation_gpu{gpu}.log',gpu) for gpu in [0,1]]
        for job in jobs:job.result()
    call('paper_analyze.py',[],'analysis.log')
    runtime_logs=OUT/'runtime_logs';runtime_logs.mkdir(exist_ok=True)
    for p in (BASE/'setup').iterdir():
        if p.is_file() and p.suffix in ['.log','.txt','.csv']:shutil.copy2(p,runtime_logs/p.name)
    save('execution_status.json',{'stage':'experiments_and_analysis_complete','status':'awaiting_verified_download_then_shutdown','time':time.time()})
    archive_manifest()

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('execution_status.json',{'status':'failed','phase':'finalization','error':str(e),'time':time.time()});raise
