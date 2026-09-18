"""Compact current-state report for the authorized experiment monitor."""
from pathlib import Path
import json,os,subprocess,time
WORK=Path(os.environ.get('R2_WORK','/root/autodl-tmp/snn_revision_round3'))
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None
queue=read(WORK/'results/queue_status.json')
active=[]
if queue:
    for r in queue.get('running',[]):
        job=r['job']
        p=WORK/'initialization/results/independent12550/status.json' if job.get('kind')=='initial' else WORK/'results'/job['stage']/job['tag']/'progress.json'
        active.append({'job':job,'progress':read(p)})
counts={stage:len(list((WORK/'results'/stage).glob('*/complete.json'))) for stage in ['screen','formal','formal_B']}
result={'time':time.time(),'queue':queue,'active':active,'completed_trajectories':counts,
        'initialization':read(WORK/'initialization/results/independent12550/complete.json') or read(WORK/'initialization/results/independent12550/status.json'),
        'scoring_completed':len([p for p in (WORK/'results/scoring').glob('*/complete.json') if not p.parent.name.startswith('training_probe')]),
        'selection':read(WORK/'results/selection.json'),
        'analysis':read(WORK/'analysis/summary.json'),
        'download_inventory_ready':(WORK/'results/download_inventory.json').exists(),
        'gpu_complete':read(WORK/'results/GPU_COMPLETE.json'),
        'queue_failure':read(WORK/'results/queue_failure.json')}
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True)
result['gpu']=gpu.stdout.strip()
print(json.dumps(result,ensure_ascii=False))
