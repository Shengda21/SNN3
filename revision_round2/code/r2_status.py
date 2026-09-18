"""Compact progress without using test outcomes to steer the queue."""
from pathlib import Path
import json,os,subprocess,time
root=Path(os.environ.get('R2_WORK','/root/snn_revision_round2'))
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
result={'time':time.time(),'training':read(root/'results/queue_status.json'),'finish':read(root/'results/finish_status.json')}
for stage in ['screen','formal']:
    rows=[]
    for folder in sorted((root/'results'/stage).glob('*')):
        if not folder.is_dir():continue
        done=read(folder/'complete.json');progress=read(folder/'progress.json');failure=read(folder/'failure.json')
        rows.append({'run':folder.name,'complete':bool(done),'step':done['step'] if done else progress.get('step',0) if progress else 0,
             'failure':failure.get('error','')[-800:] if failure and not done else None})
    result[stage]=rows
result['scoring_complete']=len(list((root/'results/scoring').glob('*/complete.json')))
result['scoring_failures']=[str(p.parent.name) for p in (root/'results/scoring').glob('*/failure.json') if not (p.parent/'complete.json').exists()]
result['timing_sessions_complete']=len(list((root/'results/timing').glob('*/complete.json')))
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print(json.dumps(result,indent=2))
