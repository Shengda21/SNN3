"""Local source and execution identity for new reproductions."""
import os,json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_sources():
 for name,h in read(B/'PACKAGE_FREEZE.json')['files'].items():assert sha(B/name)==h,name
 return sha(B/'PACKAGE_FREEZE.json')
def require_production(require_cross_gpu=False):
 verify_sources();c=read(os.environ['SNN3_REPRODUCTION_CONFIG']);assert c['execute'] is True and Path(c['code'])==B;deadline_remaining({'deadline_UTC':c['deadline_UTC']})
 return {'stage':'reader_replication','source_freeze_sha256':verify_sources(),'deadline_UTC':c['deadline_UTC'],'_runtime_policy':{'deadline_UTC':c['deadline_UTC']}}
def deadline_remaining(policy):
 import datetime,time
 left=datetime.datetime.fromisoformat(policy['deadline_UTC'].replace('Z','+00:00')).timestamp()-time.time();assert left>0;return left
def is_software_fixture(root):return False
