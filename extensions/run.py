"""Construct, adapt, fit and evaluate the additional image experiments on a local GPU."""
import argparse,ast,datetime,hashlib,importlib.util,itertools,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,default=str,allow_nan=False)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def initialize(a):
 work=a.work.resolve();assert not work.exists(),'Use a fresh work directory; completed and failed outputs are retained.';work.mkdir(parents=True);code=work/'code';shutil.copytree(HERE/'code'/a.dataset,code)
 shutil.copyfile(HERE/'portable_authority.py',code/'authority.py')
 protocol=read(code/'constructor/protocol.json');data=a.data.resolve()
 if not a.dataset.startswith('cifar'):
  receipt=read(data/'training_data_receipt.json');assert receipt['counts']=={k:protocol['data'][k] for k in ('train','validation','test')} if 'counts' in receipt else receipt['train_samples']==protocol['data']['train']
  for k in ('mean','std'):
   assert max(abs(x-y) for x,y in zip(receipt[k],protocol['normalization'][k]))<1e-10,'Training preprocessing differs from the published protocol.'
  protocol['normalization'].update(receipt)
  protocol['training_receipt_sha256']=sha(data/'training_data_receipt.json')
  if 'source_manifest_sha256' in receipt:protocol['data_manifest_sha256']=receipt['source_manifest_sha256']
  (code/'constructor/protocol.json').write_text(json.dumps(protocol,indent=2))
 # The scientific code is frozen after rebinding only local paths and data receipts.
 method=code/'method';tree=ast.parse((method/'port_common.py').read_text());schema=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEMA' for t in n.targets))
 write(method/'SOURCE_FREEZE.json',{'schema':schema,'files':{p.name:sha(p) for p in sorted(method.glob('*.py'))}})
 files={p.relative_to(code).as_posix():sha(p) for p in sorted(code.rglob('*')) if p.is_file() and '__pycache__' not in p.parts};write(code/'PACKAGE_FREEZE.json',{'files':files})
 c={'dataset':a.dataset,'code':str(code),'data':str(data),'construction_parent':str(work/'construction'),'output':str(work/'replication'),'execute':True,'gpu':a.gpu,'deadline_UTC':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(days=a.days)).isoformat()};write(work/'run.json',c)
 print(json.dumps({'status':'configured_no_GPU_execution','config':str(work/'run.json'),'source_sha256':sha(code/'PACKAGE_FREEZE.json')},indent=2))
def setup(a):
 c=read(a.work/'run.json');B=Path(c['code']);os.environ['SNN3_REPRODUCTION_CONFIG']=str((a.work/'run.json').resolve());os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu if a.gpu is not None else c['gpu']);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 site=Path(sys.executable).parent.parent/'lib/python3.12/site-packages/nvidia'
 if site.exists():os.environ.update(CUDA_PATH=str(site/'cuda_runtime'),LD_LIBRARY_PATH=str(site/'cuda_nvrtc/lib')+':'+str(site/'cuda_runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
 sys.path.insert(0,str(B));sys.path.insert(0,str(B/'method'));import authority;authority.verify_sources()
 spec=read(B/'replication.json');protocol=read(B/'constructor/protocol.json');out=Path(c['output']);out.mkdir(exist_ok=True);gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
 return c,B,spec,protocol,out,gpu
def stage(c,B,out,gpu,name,script,args,terminal):
 measure=load(B/'method/measured_queue.py','local_board_measurement');folder=out/'measurement';folder.mkdir(exist_ok=True)
 command=[sys.executable,str(HERE/'entry.py'),'--config',str(Path(c['code']).parent/'run.json'),'--script',script,'--',*map(str,args)]
 left=(datetime.datetime.fromisoformat(c['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert left>0
 measure.run_stage({'id':name,'argv':command,'cwd':str(HERE),'timeout_seconds':left,'terminal_files':[str(terminal)]},folder,gpu,1.0)
 assert read(terminal)['status'] in ('complete','passed','acceptance_only')
def campaign(c,B,spec,protocol,out,gpu):
 original=load(B/'method/controller.py','retained_campaign');outer=stage
 class Local(original.Campaign):
  def stage(self,name,script,args,terminal,wait=True):outer(c,B,out,gpu,name,script,args,terminal)
  def grid_barrier(self):pass
  def snapshot(self,*args,**kwargs):pass
 plan={'output':str(out),'constructor_bundle':str(B/'constructor'),'construction_parent':c['construction_parent'],'python':sys.executable}
 x=Local(plan);x.scored={};return x
def fixed_inputs(c,B,spec,out):
 import port_common as pc
 h=pc.source_hash() if hasattr(pc,'source_hash') else sha(B/'method/SOURCE_FREEZE.json');parent=Path(c['construction_parent']);schema=pc.SCHEMA
 if not(out/'rate_selection.json').exists():
  write(out/'rate_selection.json',{'status':'complete','schema':schema,'source_freeze_sha256':h,'construction_root':str(parent/'development'),'paired_seeds':spec['development_seeds'],'selection_uses_test':False,'choices':{m:{'rate':r} for m,r in spec['rates'].items()},'evidence':[],'execution_basis':'published fixed rates, not new development measurements','protocol_sha256':sha(B/'replication.json')})
  write(out/'development_gate.json',{'schema':schema,'source_freeze_sha256':h,'construction_root':str(parent/'development'),'paired_seeds':spec['development_seeds'],'gate_passed':spec.get('original_development_gate_passed'),'execution_basis':'published_fixed_replication','protocol_sha256':sha(B/'replication.json'),'note':'This is a fixed execution protocol, not a newly measured development result. No selection or repair claim is made.'})
def replicate(a,c,B,spec,protocol,out,gpu):
 fixed_inputs(c,B,spec,out);x=campaign(c,B,spec,protocol,out,gpu);seed=spec['formal_seeds'][a.role]
 baselines={m:x.native(a.role,seed,m,spec['rates'][m],0,'formal') for m in ('native2','native4','mtt')};candidate=x.conditional(a.role,seed);s0=x.s0(a.role);records=[s0,candidate,*baselines.values()]
 rows=x.matrix('formal'+str(seed),s0,baselines['native2'],candidate,(baselines['native4'],baselines['mtt']))
 assert len(x.scored)==52;write(out/'roles'/a.role/'records.json',records);write(out/'roles'/a.role/'validation_rows.json',rows);write(out/'roles'/a.role/'validation_complete.json',{'status':'complete','scored':{k:str(v) for k,v in x.scored.items()}})
def conditional(a,c,B,spec,protocol,out,gpu):
 assert spec.get('execution_scope')=='conditional_and_fit_only'
 fixed_inputs(c,B,spec,out);x=campaign(c,B,spec,protocol,out,gpu);seed=spec['formal_seeds'][a.role];candidate=x.conditional(a.role,seed);records=[x.s0(a.role),candidate];rows=[]
 for r in records:
  for k,t,b in itertools.product(r['checkpoints'],(2,4),('raw','matched')):
   path=x.score(r,k,t,b);rows.append(read(path))
 assert len(x.scored)==16;write(out/'roles'/a.role/'records.json',records);write(out/'roles'/a.role/'validation_rows.json',rows);write(out/'roles'/a.role/'validation_complete.json',{'status':'complete','scored':{k:str(v) for k,v in x.scored.items()}})
def freeze(c,B,spec,out):
 import port_common as pc
 from gate import bound_assess
 rows=[];checkpoints=[];routes=[]
 for role in spec['formal_seeds']:
  assert read(out/'roles'/role/'validation_complete.json')['status']=='complete';rows+=read(out/'roles'/role/'validation_rows.json')
  for r in read(out/'roles'/role/'records.json'):
   variant=r['method'];method='S0' if variant=='S0' else 'conditional' if variant=='conditional' else 'native'
   for k,path in r['checkpoints'].items():
    pc.verify_checkpoint(r['root'],path,method);checkpoints.append({'path':str(Path(path).resolve()),'sha256':sha(path),'construction_root':str(r['root']),'method':method,'variant':variant,'K':int(k)})
    for t,b in itertools.product((2,4),('raw','matched')):routes.append({'checkpoint':str(Path(path).resolve()),'construction_root':str(r['root']),'method':method,'window':t,'policy':b,'output':str(out/'scores'/f'test_{r["id"]}_k{k}_t{t}_{b}')})
 expected=12 if spec.get('execution_scope')=='conditional_and_fit_only' else 39
 assert len(checkpoints)==len({x['path'] for x in checkpoints})==expected and len(routes)==expected*4
 write(out/'formal_rows.json',rows)
 if expected==39:g=bound_assess(out/'formal_rows.json',['formal'+str(s) for s in spec['formal_seeds'].values()],'formal',Path(c['construction_parent']))
 else:g={'gate_passed':None,'status':'conditional_execution_only','note':'This entrypoint reproduces construction, conditional training, fitting and predictions. Original baseline comparisons are separate R2 execution inputs; no new comparative gate is asserted.'}
 write(out/'formal_gate.json',g)
 write(out/'FINAL_TEST_FREEZE.json',{'checkpoints':checkpoints,'routes':routes,'protocol_sha256':sha(B/'replication.json'),'formal_gate_passed':g['gate_passed'],'original_development_gate_passed':spec.get('original_development_gate_passed'),'analysis_rules':'report every configured method, start, budget, window and BN policy; no test selection'})
def test(a,c,B,spec,protocol,out,gpu):
 freeze=read(out/'FINAL_TEST_FREEZE.json');expected=12 if spec.get('execution_scope')=='conditional_and_fit_only' else 39;assert len(freeze['checkpoints'])==expected
 x=campaign(c,B,spec,protocol,out,gpu);records=read(out/'roles'/a.role/'records.json')
 for r in records:r['root']=Path(r['root']);r['checkpoints']={int(k):Path(p) for k,p in r['checkpoints'].items()}
 if expected==39:
  s0,candidate,n2,n4,mtt=records;x.matrix('formal'+str(spec['formal_seeds'][a.role]),s0,n2,candidate,(n4,mtt),split='test');assert len(x.scored)==52
 else:
  for r in records:
   for k,t,b in itertools.product(r['checkpoints'],(2,4),('raw','matched')):x.score(r,k,t,b,split='test')
  assert len(x.scored)==16
 write(out/'roles'/a.role/'test_complete.json',{'status':'complete','scored':{k:str(v) for k,v in x.scored.items()}})
def analyze(c,B,spec,out):
 records=[]
 for role in spec['formal_seeds']:
  for split in ('validation','test'):
   for path in read(out/'roles'/role/(split+'_complete.json'))['scored'].values():records.append({'path':str(path),'sha256':sha(path),'row':read(path)})
 expected=96 if spec.get('execution_scope')=='conditional_and_fit_only' else 312
 assert len(records)==expected and len(list((out/'test_route_claims').glob('*.json')))==expected//2;write(out/'all_score_receipts.json',records)
 from analyze import analyze
 analyze(out/'all_score_receipts.json',out/'analysis',outcome='scientific_complete');write(out/'complete.json',{'status':'complete_fixed_method_replication' if expected==312 else 'complete_conditional_execution','trajectories':12 if expected==312 else 3,'checkpoints':39 if expected==312 else 12,'validation_routes':expected//2,'test_routes':expected//2,'formal_gate_passed':read(out/'formal_gate.json')['gate_passed'],'original_development_gate_passed':spec.get('original_development_gate_passed'),'completion_does_not_assert_gate_pass':True})
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','construct','replicate','conditional','freeze-test','test','analyze']);p.add_argument('--work',type=Path,required=True);p.add_argument('--dataset',choices=['cifar10','bloodmnist64','eurosat','plantwild','cifar100_resformer','cifar100_spikformer']);p.add_argument('--data',type=Path);p.add_argument('--gpu',type=int,default=None);p.add_argument('--role',choices=['development','formal1','formal2','formal3']);p.add_argument('--days',type=int,default=14);p.add_argument('--execute',action='store_true');a=p.parse_args();a.work=a.work.resolve()
 if a.command=='init':assert a.dataset and a.data is not None;a.gpu=0 if a.gpu is None else a.gpu;initialize(a);return
 c,B,spec,protocol,out,gpu=setup(a)
 if a.command in ('construct','replicate','conditional','test'):assert a.execute,'GPU execution requires --execute';assert a.role is not None
 if a.command=='construct':
  root=Path(c['construction_parent'])/a.role;terminal=root/'construction'/spec.get('constructor_cohort','C-conf')/'complete.json';stage(c,B,out,gpu,'construct_'+a.role,'construct.py',['--root',root,'--data',c['data'],'--role',a.role],terminal)
 elif a.command=='replicate':assert a.role!='development' and spec.get('execution_scope')!='conditional_and_fit_only';replicate(a,c,B,spec,protocol,out,gpu)
 elif a.command=='conditional':assert a.role!='development';conditional(a,c,B,spec,protocol,out,gpu)
 elif a.command=='freeze-test':freeze(c,B,spec,out)
 elif a.command=='test':assert a.role!='development';test(a,c,B,spec,protocol,out,gpu)
 elif a.command=='analyze':analyze(c,B,spec,out)
if __name__=='__main__':main()
