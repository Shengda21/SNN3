"""Finite validation-selected R3 adaptation queue with resumable process scheduling."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

WORK=Path(os.environ['R2_WORK'])
PY=sys.executable
SCRIPT=WORK/'code/r3_train.py'
RATES=[1.25e-6,2.5e-6,5e-6,2e-5,8e-5]
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def make_job(stage,T,lr,seed,initial=None):
    row={'stage':stage,'T':T,'lr':lr,'seed':seed,'steps':2048,'tag':f's{seed}_T{T}_lr{lr:g}'}
    if initial:row['initial']=str(initial)
    return row
def run(jobs,concurrency):
    pending=list(jobs);running={};done=[]
    while pending or running:
        while pending and len(running)<concurrency:
            job=pending.pop(0)
            is_initial=job.get('kind')=='initial'
            final=(WORK/'initialization/results/independent12550/complete.json') if is_initial else WORK/'results'/job['stage']/job['tag']/'complete.json'
            if final.exists():
                status=json.loads(final.read_text())
                if status.get('status')=='complete' and (is_initial or status['step']==job['steps']):
                    done.append(job);continue
            log_path=WORK/'logs'/f"{job['stage']}_{job['tag']}.log"
            log=log_path.open('a',buffering=1)
            if is_initial:cmd=[PY,str(WORK/'code/r3_initial.py')]
            else:
                cmd=[PY,str(SCRIPT),'--workers','6','--microbatch','64']
                for key,value in job.items():cmd.extend(['--'+key,str(value)])
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            running[child.pid]=(child,log,job,time.time())
        for pid,(child,log,job,started) in list(running.items()):
            rc=child.poll()
            if rc is None:continue
            log.close();del running[pid]
            if rc:
                failure={'status':'failed','job':job,'returncode':rc,'other_running_pids':list(running),'at':time.time()}
                write(WORK/'results/queue_failure.json',failure)
                # Preserve other independent useful trajectories rather than killing them.
                for other,other_log,_,_ in running.values():other.wait();other_log.close()
                raise RuntimeError(json.dumps(failure))
            done.append(job)
        state={'status':'running' if pending or running else 'complete','completed':len(done),'total':len(jobs),
               'running':[{'pid':pid,'job':entry[2],'started':entry[3]} for pid,entry in running.items()],
               'pending':len(pending),'updated':time.time(),'concurrency':concurrency}
        write(WORK/'results/queue_status.json',state)
        if pending or running:time.sleep(5)
    return done
def select():
    candidates=[]
    for T in [2,4]:
        for lr in RATES:
            values=[]
            for seed in [9611,9612]:
                job=make_job('screen',T,lr,seed)
                record=json.loads((WORK/'results/screen'/job['tag']/'complete.json').read_text())
                e=next(x for x in record['evaluations'] if x['step']==2048)
                assert e['validation']['split']=='validation' and e['validation']['samples']==5000
                values.append({'seed':seed,'ce':e['validation']['ce'],'accuracy':e['validation']['accuracy']})
            candidates.append({'T':T,'lr':lr,'seeds':values,'mean_ce':sum(x['ce'] for x in values)/2})
    chosen={str(T):min((c for c in candidates if c['T']==T),key=lambda x:(x['mean_ce'],x['lr']))['lr'] for T in [2,4]}
    result={'selected':chosen,'candidates':candidates,'rule':'minimum mean native raw validation CE at 2048; exact tie smaller LR','uses_test':False,'time':time.time()}
    write(WORK/'results/selection.json',result);return chosen
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','formal','second','all']);ap.add_argument('--concurrency',type=int,default=2)
    args=ap.parse_args()
    if args.stage in ['screen','all']:
        jobs=[{'kind':'initial','stage':'initialization','tag':'independent12550'}]+[make_job('screen',T,lr,seed) for lr in RATES for seed in [9611,9612] for T in [2,4]]
        write(WORK/'protocol/screen_queue.json',jobs);run(jobs,args.concurrency);select()
    if args.stage in ['formal','all']:
        chosen=select();jobs=[]
        for seed in range(9621,9626):
            for T in [2,4]:
                for lr in sorted(set([chosen[str(T)],2e-5])):jobs.append(make_job('formal',T,lr,seed))
        write(WORK/'protocol/formal_queue.json',jobs);run(jobs,args.concurrency)
    if args.stage in ['second','all']:
        initial=WORK/'initialization/results/independent12550/best.pt'
        if not initial.exists():raise FileNotFoundError(initial)
        chosen=select()
        jobs=[make_job('formal_B',T,chosen[str(T)],seed,initial) for seed in range(9721,9724) for T in [2,4]]
        write(WORK/'protocol/second_queue.json',jobs);run(jobs,args.concurrency)
