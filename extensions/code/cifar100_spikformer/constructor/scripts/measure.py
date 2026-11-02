"""C-conf inference/calibration timing, same fresh-process protocol as current Table 7."""
import argparse
from pathlib import Path
import time
import numpy as np
from common import BUNDLE, environment, read, sha, verify_file, verify_prepared, write


def _worker(root,row_path,output):
    root=verify_prepared(root)
    environment(root)
    row=read(row_path)
    verify_file(row['checkpoint'],row['checkpoint_sha256'])
    if output.exists():
        assert read(output)['row']==row
        return
    import torch
    import runtime as rt
    from normalization import calibrate_bn
    import sys
    sys.path.insert(0,str(root/'legacy/code'))
    from r2_score import capture,training_inputs
    rt.configure()
    tick=time.perf_counter()
    model=rt.load_model(root,row['architecture'],row['checkpoint'],row['infer_T'])
    model.requires_grad_(False)
    model_load_seconds=time.perf_counter()-tick
    cal_seconds,cal_input_seconds=None,None
    if row['calibration']!='raw':
        tick=time.perf_counter()
        images,_=rt.calibration_images(root,9640)
        cal_input_seconds=time.perf_counter()-tick
        torch.cuda.synchronize();tick=time.perf_counter()
        _,checks=calibrate_bn(model,images,lambda:rt.reset(model),lambda x:rt.forward(model,x),batch=64)
        torch.cuda.synchronize();cal_seconds=time.perf_counter()-tick
        assert checks['protected_values_and_dtypes_unchanged']
        rt.prepare_mode(model,False)
    samples=training_inputs('vision')[0]
    def forward(batch):
        return rt.forward(model,batch)
    initial=samples[:4].cuda().clone()
    for _ in range(5):forward(initial)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();forward(initial);torch.cuda.synchronize()
    eager_peak=torch.cuda.max_memory_allocated()/2**30
    del initial
    inp,graph,logits,checks=capture(forward,samples)
    engines=['eager','graph'] if all(r['bitwise_equal'] for r in checks['ABA']) else ['eager']
    values=[]
    for round_index in range(3):
        for engine in (engines if round_index%2==0 else list(reversed(engines))):
            for _ in range(5):
                graph.replay() if engine=='graph' else forward(inp)
            for repeat in range(10):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                graph.replay() if engine=='graph' else forward(inp)
                torch.cuda.synchronize()
                values.append({'engine':engine,'round':round_index,'repeat':repeat,
                    'wall_ms':1000*(time.perf_counter()-tick),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'reserved_gib':torch.cuda.memory_reserved()/2**30})
    write(output,{'status':'complete','row':row,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'calibration_seconds':cal_seconds,'calibration_input_prepare_seconds':cal_input_seconds,
        'model_load_seconds':model_load_seconds,'batch':4,'copy_scope':'GPU-resident forward; host transfer excluded',
        'eager_standalone_peak_allocated_gib':eager_peak,'graph_checks':checks,'measurements':values,
        'warmup_capture_seconds':checks['warmup_capture_seconds'],
        'medians_ms':{engine:float(np.median([r['wall_ms'] for r in values if r['engine']==engine])) for engine in engines},
        'repeat_scope':'one fixed trained source; process/call repeats are not adaptation seeds'})


def worker(root,row_path,output):
    import torch
    # Both eager and graph routes use the same no-grad state, including warmups and capture.
    with torch.no_grad():
        return _worker(root,row_path,output)


def run(root,command):
    rows=read(root/'protocol/scoring_queue.json')
    chosen=[r for r in rows if r['cohort']=='C-conf' and r['updates'] in (0,2048)
            and r['seed'] in (None,11021)
            and (r['calibration']=='raw' or r['cal_T']==r['infer_T'])]
    assert chosen
    directory=root/'timing'
    directory.mkdir(exist_ok=True)
    for row in chosen:
        source=directory/f'{row["id"]}_input.json'
        write(source,row)
        for process in range(3):
            output=directory/f'{row["id"]}_process{process}.json'
            command(root,f'timing_{row["id"]}_p{process}','measure.py','--row',source,'--output',output)
    write(directory/'measurement_scope.json',{'configurations':len(chosen),'processes_per_configuration':3,
        'training_seed':11021,'warmup_calls_before_capture':5,'warmup_calls_before_each_timing_block':5,
        'measured_calls_per_route_process':30,'quality_scoring_covers_three_adaptation_seeds':True,
        'C_only':True,'F_timing_not_claimed':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--row',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    worker(a.root,a.row,a.output)
