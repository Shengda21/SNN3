"""Compare required assets with identities recorded in the preceding experiment."""
import hashlib,json,os,platform,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(os.environ.get('R2_ARCHIVE','/root/autodl-tmp/snn0907'));work=Path(os.environ.get('R2_WORK','/root/snn_revision_round2'))
inv=json.loads((work/'protocol/asset_inventory.json').read_text())
def check(a):
 p=root/a['archive_relative_path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return {'path':str(p),'bytes':p.stat().st_size,'matches':h.hexdigest()==a['recorded_sha256']}
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,inv['asset_checks']))
dest=work/'results/E0';dest.mkdir(parents=True,exist_ok=True)
(dest/'asset_check.json').write_text(json.dumps(rows,indent=2))
assert all(r['matches'] for r in rows),rows
import sys,shlex
for name,cmd in [('nvidia_smi','nvidia-smi -q'),('cpu','lscpu'),('disk','df -h'),('packages',shlex.quote(sys.executable)+' -m pip freeze')]:
 p=subprocess.run(cmd,shell=True,capture_output=True,text=True)
 (dest/f'{name}.txt').write_text(p.stdout+p.stderr)
print(json.dumps({'checked':len(rows),'all_match':True,'python':platform.python_version()}))
