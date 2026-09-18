"""One fresh process per native checkpoint pair; final states and batch four only."""
import argparse,gc,json,os,subprocess,time
import numpy as np
import torch
from r2_score import WORK,configure,training_inputs,language,vision,capture,write_json


def device_snapshot():
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name',
        '--format=csv,noheader'],text=True).strip()
    others=[line for line in active.splitlines() if int(line.split(',')[0].strip())!=os.getpid()]
    assert not others, 'Other GPU compute processes are active: '+str(others)
    devices=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw',
        '--format=csv,noheader'],text=True).strip()
    return {'unix_time':time.time(),'compute_processes':active,'other_compute_processes':others,
        'device_state':devices}


@torch.no_grad()
def run(session):
    configure();folder=WORK/'results/timing'/session['id']
    if (folder/'complete.json').exists():return
    if folder.exists() and any(folder.iterdir()):
        failed=WORK/'results/timing_incomplete_attempts'/f'{session["id"]}_{time.time_ns()}'
        failed.parent.mkdir(parents=True,exist_ok=True)
        assert folder.resolve().is_relative_to(WORK.resolve()) and failed.resolve().is_relative_to(WORK.resolve())
        folder.rename(failed)
    folder.mkdir(parents=True,exist_ok=True)
    started_at=time.time()
    idle_check=device_snapshot()
    inputs,_=training_inputs(session['task'])
    for T in session['T_order']:
        path=folder/f'T{T}.json'
        checkpoint=session['checkpoints'][str(T)]
        if session['task']=='language':m,forward=language(T,checkpoint)
        else:m,forward,sites=vision(T,checkpoint)
        for _ in range(5):out=forward(inputs[:4].cuda())
        del out
        x,graph,z,check=capture(forward,inputs)
        x.copy_(inputs[:4]);torch.cuda.synchronize()
        def eager():return forward(x)
        def replay():graph.replay();return z
        functions={'eager':eager,'graph':replay};rows=[];sequence_record=[]
        for rnd in range(3):
            order=['eager','graph'] if (session['session']+rnd)%2==0 else ['graph','eager']
            sequence_record.append(order)
            for engine in order:
                fn=functions[engine]
                for _ in range(5):warm=fn()
                del warm
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                for rep in range(10):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    tick=time.perf_counter();begin.record()
                    output=fn();end.record();end.synchronize()
                    rows.append({'round':rnd,'repeat':rep,'engine':engine,'wall_ms':1000*(time.perf_counter()-tick),
                      'cuda_ms':begin.elapsed_time(end),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                      'reserved_gib':torch.cuda.memory_reserved()/2**30})
                    del output
        meta={'status':'complete','session':session,'T':T,'checkpoint':checkpoint,'graph_checks':check,
            'process_pid':os.getpid(),'process_session_started_at':started_at,
            'device_idle_check':idle_check,'device_state_after_measurement':device_snapshot(),
            'gpu_name':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'torch':torch.__version__,'cuda':torch.version.cuda,'threads':torch.get_num_threads(),
            'input_source':'first 4 training-side implementation inputs','copy_scope':'input preloaded on device, no copy inside measured call',
            'engine_order':sequence_record,'warmup_per_engine_sequence':5,'measurements':rows,
            'medians':{e:float(np.median([r['wall_ms'] for r in rows if r['engine']==e])) for e in functions}}
        write_json(path,meta);print(json.dumps({'id':session['id'],'T':T,'medians':meta['medians']}),flush=True)
        del functions,eager,replay,fn,graph,z,x,m,forward;gc.collect();torch.cuda.empty_cache()
    write_json(folder/'complete.json',{'status':'complete','session':session['id']})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);a=ap.parse_args()
    manifest=json.loads((WORK/'results/model_analysis_lock.json').read_text())
    session=next(s for s in manifest['timing_sessions'] if s['id']==a.id)
    assert (WORK/'results/scoring_finished.json').exists()
    run(session)
