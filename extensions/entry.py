"""Bind public local identities, then execute a complete retained scientific entrypoint."""
import argparse,os,sys,runpy,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--script',choices=['construct.py','native.py','train.py','fit_pack.py','score.py'],required=True);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();args=a.args[1:] if a.args[:1]==['--'] else a.args
c=json.loads(a.config.read_bytes());os.environ['SNN3_REPRODUCTION_CONFIG']=str(a.config.resolve());B=Path(c['code']);sys.path.insert(0,str(B));sys.path.insert(0,str(B/'method'));import authority
authority.require_production()
if a.script!='construct.py':
 import port_common as pc
 def parent(bundle,root):
  root=Path(root).resolve();bundle=Path(bundle).resolve();assert bundle==B/'constructor';authority.verify_sources();role=pc.read(root/'environment/construction_role.json');assert role['role']==root.name and role['seed']==pc.read(bundle/'protocol.json')['construction_roles'][root.name] and role['protocol_sha256']==pc.sha(bundle/'protocol.json')
  candidates=[x for x in (root/'environment').glob('*_prepared.json')];assert len(candidates)==1;prepared=candidates[0];v=pc.read(prepared);assert v['protocol_sha256']==pc.sha(bundle/'protocol.json')
  for row in v['sources']:
   assert pc.sha(bundle/row['path'])==row['sha256'];relative=row['path']
   if relative.startswith('payload/vision/'):assert pc.sha(root/'archive/direction1_budget_rotation_20260907'/relative.removeprefix('payload/vision/'))==row['sha256']
   elif relative.startswith('payload/legacy_code/'):assert pc.sha(root/'legacy/code'/relative.removeprefix('payload/legacy_code/'))==row['sha256']
  h=pc.source_hash() if hasattr(pc,'source_hash') else pc.sha(B/'method/SOURCE_FREEZE.json')
  return {'constructor_protocol_sha256':pc.sha(bundle/'protocol.json'),'construction_root':str(root),'prepared_sha256':pc.sha(prepared),'construction_role_sha256':pc.sha(root/'environment/construction_role.json'),'role':root.name,'source_freeze_sha256':h,'schema':pc.SCHEMA,'software_fixture':False}
 def gate(path,root):
  v=pc.read(path);spec=pc.read(B/'replication.json');assert v['execution_basis']=='published_fixed_replication' and v['gate_passed']==spec.get('original_development_gate_passed') and v['protocol_sha256']==pc.sha(B/'replication.json');return v
 def test_guard(path,checkpoint,root,method):
  v=pc.read(path);count=12 if pc.read(B/'replication.json').get('execution_scope')=='conditional_and_fit_only' else 39;assert v['protocol_sha256']==pc.sha(B/'replication.json') and len(v['checkpoints'])==count and len({x['path'] for x in v['checkpoints']})==count
  dest=Path(args[args.index('--out')+1]).resolve();key={'checkpoint':str(Path(checkpoint).resolve()),'construction_root':str(Path(root).resolve()),'method':method,'window':int(args[args.index('--window')+1]),'policy':args[args.index('--policy')+1],'output':str(dest)}
  assert key in v['routes']
  assert any(x['path']==key['checkpoint'] and x['sha256']==pc.sha(checkpoint) and x['construction_root']==key['construction_root'] for x in v['checkpoints'])
  claim=Path(c['output'])/'test_route_claims'/(dest.name+'.json')
  if not claim.exists():
   claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as f:json.dump({'route':key,'freeze_sha256':pc.sha(path),'single_attempt':True,'PID':os.getpid()},f)
  else:
   previous=pc.read(claim);assert previous['freeze_sha256']==pc.sha(path) and previous['route']==key and previous['PID']==os.getpid()
  return v
 pc.verify_parent=parent;pc.verify_development_gate=gate;pc.verify_test_freeze=test_guard
 script=B/'method'/a.script
else:script=B/'constructor/construct.py'
sys.argv=[str(script),*args];runpy.run_path(str(script),run_name='__main__')
