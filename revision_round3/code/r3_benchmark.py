"""Measure real-model sequential versus concurrent adaptation throughput."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from r2_vision import WORK,write_json

def launch(tag,T):
    log=(WORK/'logs'/f'benchmark_{tag}.log').open('w')
    cmd=[sys.executable,str(WORK/'code/r3_train.py'),'--T',str(T),'--seed','9698','--lr','2e-5',
         '--steps','64','--stage','probe','--tag',tag,'--workers','6']
    return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log,tag
def finish(handle):
    proc,log,tag=handle;rc=proc.wait();log.close()
    if rc:return {'tag':tag,'returncode':rc,'failed':True}
    path=WORK/'results/probe'/tag
    steps=[json.loads(l) for l in (path/'steps.jsonl').read_text().splitlines()]
    complete=json.loads((path/'complete.json').read_text())
    return {'tag':tag,'returncode':0,'median_update_seconds_after_warmup':float(np.median([x['seconds'] for x in steps[16:]])),
            'peak_allocated_gib':complete['peak_allocated_gib'],'training_seconds':complete['training_seconds']}
if __name__=='__main__':
    singles=[];tick=time.perf_counter()
    for T in [2,4]:singles.append(finish(launch(f'single_T{T}',T)))
    sequential_wall=time.perf_counter()-tick
    assert not any(r.get('failed') for r in singles),singles
    tick=time.perf_counter();handles=[launch(f'parallel_T{T}',T) for T in [2,4]]
    pairs=[finish(h) for h in handles];parallel_wall=time.perf_counter()-tick
    success=not any(r.get('failed') for r in pairs)
    improvement=sequential_wall/parallel_wall
    # Include startup/validation overhead because each finite trajectory incurs it.
    chosen=2 if success and improvement>1.05 and 2*singles[1]['peak_allocated_gib']<28.5 else 1
    result={'single':singles,'parallel':pairs,'sequential_wall_seconds':sequential_wall,
            'parallel_wall_seconds':parallel_wall,'speedup':improvement,'chosen_concurrency':chosen,
            'scientific_batch_and_precision_changed':False,'status':'passed'}
    write_json(WORK/'results/E0/throughput.json',result)
    print(json.dumps(result),flush=True)
