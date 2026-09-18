"""Run the frozen symmetric search and five final pairs on two GPUs."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
WORK=Path(os.environ.get('R2_WORK','/root/snn_revision_round2'));PY=sys.executable

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def run(gpu,stage,seed,T,lr,steps):
    tag=f's{seed}_T{T}_lr{lr:g}'
    command=[PY,str(WORK/'code/r2_vision.py'),'--T',str(T),'--seed',str(seed),'--lr',str(lr),
      '--steps',str(steps),'--tag',tag,'--stage',stage]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    cpus=sorted(os.sched_getaffinity(0));half=len(cpus)//2;allowed=cpus[:half] if gpu==0 else cpus[half:]
    command=['taskset','-c',','.join(map(str,allowed))]+command
    (WORK/'logs').mkdir(exist_ok=True)
    with (WORK/'logs'/f'{stage}_{tag}.log').open('a') as f:
        p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    result=WORK/'results'/stage/tag/'complete.json'
    row={'gpu':gpu,'tag':tag,'T':T,'seed':seed,'lr':lr,'exit_code':p.returncode,'complete_path':str(result)}
    if p.returncode==0 and result.exists():row['result']=json.loads(result.read_text())
    else:row['failure']=str(WORK/'results'/stage/tag/'failure.json')
    print(json.dumps({k:v for k,v in row.items() if k!='result'}),flush=True)
    return row

def screen_gpu(gpu):
    seed=[9611,9612][gpu];rows=[]
    for j,lr in enumerate([5e-6,2e-5,8e-5]):
        for T in ([2,4] if (j+gpu)%2==0 else [4,2]):rows.append(run(gpu,'screen',seed,T,lr,512))
    return rows

def formal_gpu(gpu,selection):
    pairs=[(9621,[2,4]),(9623,[4,2]),(9625,[2,4])] if gpu==0 else [(9622,[4,2]),(9624,[2,4])]
    return [run(gpu,'formal',seed,T,selection[str(T)],2048) for seed,order in pairs for T in order]

def main():
    assert (WORK/'results/E0/vision_migration.json').exists()
    assert all(json.loads((WORK/'results/E0/resume_check.json').read_text()).values())
    write(WORK/'results/queue_status.json',{'phase':'screening','started_at':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        screen=sum(list(pool.map(screen_gpu,[0,1])),[])
    write(WORK/'results/screening_runs.json',screen)
    selection={};candidates=[]
    for T in [2,4]:
        for lr in [5e-6,2e-5,8e-5]:
            group=[r for r in screen if r['T']==T and r['lr']==lr]
            valid=len(group)==2 and all('result' in r for r in group)
            mean=sum(r['result']['evaluations'][-1]['validation']['ce'] for r in group)/2 if valid else None
            candidates.append({'T':T,'lr':lr,'eligible':valid,'mean_validation_ce':mean,'seeds':[r['seed'] for r in group]})
        eligible=[r for r in candidates if r['T']==T and r['eligible']]
        if not eligible:
            write(WORK/'results/queue_status.json',{'phase':'no_eligible_candidate','T':T,'candidates':candidates})
            raise RuntimeError(f'No eligible T{T} candidate')
        selection[str(T)]=min(eligible,key=lambda r:(r['mean_validation_ce'],r['lr']))['lr']
    write(WORK/'results/selection.json',{'selected':selection,'candidates':candidates,'test_used':False,'selected_at':time.time()})
    write(WORK/'results/queue_status.json',{'phase':'formal','selected':selection,'time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        formal=sum(list(pool.map(lambda gpu:formal_gpu(gpu,selection),[0,1])),[])
    write(WORK/'results/formal_runs.json',formal)
    status='training_complete' if all('result' in r for r in formal) else 'training_finished_with_failures'
    write(WORK/'results/queue_status.json',{'phase':status,'time':time.time(),'formal_completed':sum('result' in r for r in formal)})
    print(status,flush=True)

if __name__=='__main__':main()
