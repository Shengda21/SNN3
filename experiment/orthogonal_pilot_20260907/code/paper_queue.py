"""Two isolated GPU training workers; same-seed T pairs stay on one GPU."""
import concurrent.futures,json,os,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/paper_experiments_20260908';OUT.mkdir(parents=True,exist_ok=True)

def save(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def run(gpu,stage,T,seed,lr,steps):
    tag=f'T{T}_seed{seed}_lr{lr:g}';out=OUT/stage/tag
    if (out/'complete.json').exists():return True
    # Adopt an already running job when the scheduler is restarted; never duplicate it.
    active=out/'active.pid'
    if active.exists():
        pid=int(active.read_text())
        while Path(f'/proc/{pid}/cmdline').exists():
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'paper_train.py train' not in command:break
            if (out/'complete.json').exists():return True
            time.sleep(5)
        if (out/'complete.json').exists():return True
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    cmd=[sys.executable,str(ROOT/'code/paper_train.py'),'train','--T',str(T),'--seed',str(seed),'--lr',str(lr),'--steps',str(steps),'--stage',stage]
    save(f'worker{gpu}.json',{'stage':stage,'T':T,'seed':seed,'lr':lr,'steps':steps,'started':time.time()})
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    with (logs/f'{stage}_{tag}.log').open('a') as f:
        result=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        out.mkdir(parents=True,exist_ok=True);active.write_text(str(result.pid));returncode=result.wait()
    active.unlink(missing_ok=True)
    okay=returncode==0 and (out/'complete.json').exists()
    print(json.dumps({'gpu':gpu,'stage':stage,'run':tag,'success':okay}),flush=True)
    return okay

def worker(gpu,stage,seeds,lrs,steps):
    results=[]
    for i,seed in enumerate(seeds):
        for lr in lrs:
            for T in ([2,4] if i%2==0 else [4,2]):
                actual=lr[T] if isinstance(lr,dict) else lr
                results.append(run(gpu,stage,T,seed,actual,steps))
    return all(results)

def formal_with_vision():
    okay=worker(1,'formal',[8622],[SELECTED],4096)
    status=OUT/'vision_status.json'
    if status.exists() and json.loads(status.read_text()).get('state') in ['ready_original','ready_reconstructed']:return okay
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',TOKENIZERS_PARALLELISM='false')
    with (OUT/'logs/vision_reconstruction.log').open('a') as f:
        result=subprocess.run([sys.executable,str(ROOT/'code/paper_vision_restore.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({'vision_reconstruction_exit':result.returncode}),flush=True)
    return okay

def main():
    gate=OUT/'E1/resume_check.json'
    assert gate.exists() and all(json.loads(gate.read_text()).values()),'E1 resume gate not complete'
    save('execution_status.json',{'stage':'screen','status':'running','start_time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,g,'screen',[8611+g],[lr],2048) for g in [0,1] for lr in [6e-6,2e-5,6e-5,2e-4]]
        flags=[f.result() for f in futures]
    selection={};all_candidates=[]
    for T in [2,4]:
        candidates=[]
        for lr in [6e-6,2e-5,6e-5,2e-4]:
            values=[]
            for seed in [8611,8612]:
                p=OUT/f'screen/T{T}_seed{seed}_lr{lr:g}/complete.json'
                if p.exists():values.append(json.loads(p.read_text())['evaluations'][-1]['tune']['ce'])
            row={'T':T,'lr':lr,'ce_each':values,'eligible':len(values)==2}
            if len(values)==2:row['mean_ce']=sum(values)/2;candidates.append(row)
            all_candidates.append(row)
        if not candidates:raise RuntimeError(f'All screening candidates failed for T{T}; investigate before extending protocol')
        winner=min(candidates,key=lambda r:(r['mean_ce'],r['lr']));selection[T]=winner['lr']
    save('lr_selection.json',{'selected':selection,'candidates':all_candidates,'screen_all_jobs_succeeded':all(flags),'rule':'minimum two-seed mean tune CE at 2048; exact tie lower lr','final_confirmation_accessed':False})
    save('execution_status.json',{'stage':'formal','status':'running','selected_lr':selection})
    global SELECTED
    SELECTED=selection
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,'formal',[8621,8623],[selection],4096),pool.submit(formal_with_vision)]
        flags=[f.result() for f in futures]
    if not all(flags):raise RuntimeError('Formal trajectory failed; retained failure logs and latest state')
    # Rebuild unavailable historical language trajectories, clearly labeled.
    save('execution_status.json',{'stage':'legacy_reconstruction','status':'running'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,'legacy',[7621,7623],[{2:6e-5,4:2e-5}],512),pool.submit(worker,1,'legacy',[7622],[{2:6e-5,4:2e-5}],512)]
        flags=[f.result() for f in futures]
    save('training_finished.json',{'formal_complete':True,'legacy_reconstruction_complete':all(flags),'time':time.time(),'final_confirmation_accessed':False})
    save('execution_status.json',{'stage':'training_finished','status':'awaiting_benchmark_and_locked_confirmation','final_confirmation_accessed':False})

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('execution_status.json',{'status':'failed','error':str(e),'type':type(e).__name__,'time':time.time()});raise
