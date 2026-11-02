"""Finite, fail-closed process/board-energy measurement; no scientific selection."""
import argparse, csv, hashlib, json, os, pathlib, subprocess, threading, time

def write(path, value):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def digest(path):
    with open(path, 'rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def query(gpu, fields):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=' + ','.join(fields),
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    return [part.strip() for part in next(csv.reader(result.stdout.splitlines()))]

def run_stage(stage, out, gpu, interval):
    directory = out / stage['id']
    if directory.exists(): raise RuntimeError('Preserve existing stage, do not silently rerun: ' + str(directory))
    directory.mkdir()
    for item in stage.get('requires', []):
        path = pathlib.Path(item['path'])
        if not path.is_file() or (item.get('sha256') and digest(path) != item['sha256']):
            raise RuntimeError('Missing/mismatched dependency: ' + str(path))
    for path in stage.get('must_not_exist', []):
        if pathlib.Path(path).exists(): raise RuntimeError('Would reuse old scientific result: ' + path)
    existing = subprocess.run(['nvidia-smi','-i',str(gpu),
        '--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    if existing: raise RuntimeError('Resource measurement requires an idle GPU before stage: '+existing)
    samples, errors, stop = [], [], threading.Event()
    def sample():
        try:
            before = time.monotonic()
            watts, memory, util = query(gpu, ['power.draw','memory.used','utilization.gpu'])
            after = time.monotonic()
            samples.append({'monotonic': (before+after)/2, 'unix':time.time(),
                'watts':float(watts),'device_memory_MiB':float(memory), 'util_percent':float(util),
                'query_seconds':after-before})
        except Exception as error: errors.append(repr(error))
    sample()
    if errors: raise RuntimeError('Power measurement unavailable: ' + str(errors))
    def sampler():
        while not stop.wait(interval): sample()
    thread = threading.Thread(target=sampler, daemon=True); thread.start()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)); env.update(stage.get('env', {}))
    if env['CUDA_VISIBLE_DEVICES'] != str(gpu): raise RuntimeError('GPU remapping forbidden')
    started = time.monotonic(); write(directory/'started.json', {'stage':stage,'pid':os.getpid(),'unix':time.time()})
    try:
        with open(directory/'process.log','wb') as log:
            child = subprocess.Popen(stage['argv'], cwd=stage['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            write(directory/'pid.json', {'pid':child.pid,'gpu':gpu})
            try: code = child.wait(timeout=stage['timeout_seconds'])
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                code = -124
    finally:
        ended = time.monotonic(); stop.set(); thread.join(timeout=20); sample()
        write(directory/'power_samples.json',samples)
    # Clip integration to actual subprocess interval; bracketing samples permit interpolation.
    energy = 0.; gaps = []
    for a,b in zip(samples,samples[1:]):
        lo, hi = max(started,a['monotonic']), min(ended,b['monotonic'])
        dt=b['monotonic']-a['monotonic']; gaps.append(dt)
        if hi > lo and dt>0:
            power=lambda t:a['watts']+(b['watts']-a['watts'])*(t-a['monotonic'])/dt
            energy += (power(lo)+power(hi))*.5*(hi-lo)
    complete = code == 0 and not errors and len(samples)>1 and max(gaps,default=999)<max(5,interval*5)
    terminals=[]
    for path in stage.get('terminal_files',[]):
        exists=pathlib.Path(path).is_file(); complete &= exists
        terminals.append({'path':path,'sha256':digest(path) if exists else None})
    receipt={'status':'complete' if complete else 'failed','returncode':code,
        'process_wall_seconds':ended-started,'board_energy_J':energy,
        'energy_valid':not errors and len(samples)>1 and max(gaps,default=999)<max(5,interval*5),
        'power_errors':errors,'max_sample_gap_seconds':max(gaps,default=None),
        'device_peak_memory_MiB':max(s['device_memory_MiB'] for s in samples),
        'memory_scope':'whole-board sampled usage; distinct from child PyTorch allocator peak',
        'energy_scope':'new RTX5090 board power, idle included during process; not historical or whole-system energy',
        'terminal_files':terminals,'stage':stage}
    write(directory/'receipt.json',receipt)
    if not complete: raise RuntimeError('Stage failed; preserve and stop: '+stage['id'])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); plan=json.loads(args.plan.read_text(encoding='utf-8'))
    assert plan['physical_gpu']==0 and plan['selection_uses_test'] is False
    assert plan['historical_energy_reconstruction'] is False
    assert plan['stages'] and len({s['id'] for s in plan['stages']})==len(plan['stages'])
    for stage in plan['stages']:
        assert isinstance(stage['argv'],list) and stage['timeout_seconds']>0
    if not args.execute:
        print(json.dumps({'status':'validated_only','stages':len(plan['stages']),'plan_sha256':digest(args.plan)})); return
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'frozen_plan.json',plan)
    write(args.output/'environment.json',{'hardware':query(0,['name','uuid','driver_version','memory.total']),
        'python':subprocess.check_output([plan['python'],'-c','import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)'],text=True),
        'plan_sha256':digest(args.plan),'construction_scope':plan['construction_scope']})
    try:
        for stage in plan['stages']: run_stage(stage,args.output,0,plan.get('sample_interval_seconds',.5))
        write(args.output/'complete.json',{'status':'complete','stages':len(plan['stages'])})
    except Exception as error:
        write(args.output/'failed.json',{'status':'failed','error':repr(error)}); raise

if __name__=='__main__': main()
