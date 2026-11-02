"""Run an unchanged frozen Python entrypoint with synchronized calibration timings.

Use as python instrument_entry.py --receipt X --script original.py -- original args.
The instrumentation changes no model parameters, data selection or hyperparameters.
"""
import argparse, importlib, json, pathlib, runpy, sys, time

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--receipt',required=True)
    parser.add_argument('--script',required=True); parser.add_argument('args',nargs=argparse.REMAINDER)
    a=parser.parse_args(); script=pathlib.Path(a.script).resolve(); sys.path.insert(0,str(script.parent))
    import torch
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    calibration=[]
    # R2 runtime functions are importable without training. Other workflows keep their
    # own frozen timing receipts; absence of runtime calibration is explicitly recorded.
    try: runtime=importlib.import_module('runtime')
    except ModuleNotFoundError as error:
        if error.name != 'runtime': raise
        runtime=None
    if runtime is not None:
        for name in ('calibrate','evaluate_validation'):
            original=getattr(runtime,name,None)
            if original is None: continue
            def wrap(*args,_name=name,_original=original,**kwargs):
                torch.cuda.synchronize(); start=time.monotonic()
                try: return _original(*args,**kwargs)
                finally:
                    torch.cuda.synchronize(); calibration.append({'function':_name,'wall_seconds':time.monotonic()-start})
            setattr(runtime,name,wrap)
    sys.argv=[str(script)]+(a.args[1:] if a.args[:1]==['--'] else a.args)
    start=time.monotonic(); status='failed'
    try: runpy.run_path(str(script),run_name='__main__'); status='complete'
    finally:
        torch.cuda.synchronize()
        value={'status':status,'entry_wall_seconds':time.monotonic()-start,
            'calibration_and_validation_calls':calibration,
            'observed_peak_allocated_bytes_since_most_recent_reset':torch.cuda.max_memory_allocated(),
            'observed_peak_reserved_bytes_since_most_recent_reset':torch.cuda.max_memory_reserved(),
            'memory_observation_scope':'entrypoint-final CUDA allocator counters since the most recent reset_peak_memory_stats call; inner training code may reset counters, so these values are not whole-process maxima; subprocesses require separate receipts',
            'runtime_instrumented':runtime is not None}
        dest=pathlib.Path(a.receipt); dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(value,indent=2),encoding='utf-8')

if __name__=='__main__': main()
