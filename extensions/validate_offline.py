"""Check all six configurations and the real score-packet naming contract without a GPU."""
import ast,hashlib,importlib.util,itertools,json,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 reports=[];total=0
 for folder in sorted((HERE/'code').iterdir()):
  if not folder.is_dir():continue
  spec=json.loads((folder/'replication.json').read_bytes());p=json.loads((folder/'constructor/protocol.json').read_bytes());assert spec['updates']==2048 and spec['budgets']==[512,1024,2048]
  assert len(spec['formal_seeds'])==3 and len(set(spec['formal_seeds'].values()))==3
  cohort=spec.get('constructor_cohort','C-conf');assert p['cohorts'][cohort]['construction_epochs']==(30 if folder.name=='cifar100_spikformer' else 600)
  constructor=ast.parse((folder/'constructor/construct.py').read_text(encoding='utf8'));calls=[n for n in ast.walk(constructor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='train_job' and n.func.attr=='construct'];assert len(calls)==1 and isinstance(calls[0].args[1],ast.Constant) and calls[0].args[1].value==cohort
  adapter=folder/'constructor/scripts/mtt_adapter.py';s=importlib.util.spec_from_file_location('one_offline_mtt',adapter);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.local_table_check()
  # Execute the retained Campaign.score implementation, replacing only dispatch.
  tree=ast.parse((folder/'method/controller.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Campaign')
  with tempfile.TemporaryDirectory() as temporary:
   ns={'Path':Path,'BUDGETS':(512,1024,2048)};exec(compile(ast.Module(body=[node],type_ignores=[]),'<retained Campaign>','exec'),ns)
   out=Path(temporary)/'output';x=ns['Campaign']({'output':str(out),'constructor_bundle':str(folder/'constructor'),'construction_parent':str(Path(temporary)/'construction')});x.scored={};packets=[]
   def capture(name,script,args,terminal,**kwargs):packets.append((name,script,list(map(str,args)),str(terminal)))
   x.stage=capture
   for role in spec['formal_seeds']:assert x.s0(role)['checkpoints'][0]==Path(temporary)/'construction'/role/'construction'/cohort/'best.pt'
   for role,seed in spec['formal_seeds'].items():
    variants=('S0','conditional') if spec.get('execution_scope') else ('S0','conditional','native2','native4','mtt')
    for variant in variants:
     ident=role+'_S0' if variant=='S0' else f'{role}_conditional_s{seed}' if variant=='conditional' else f'formal_{variant}_s{seed}_r00';ks=(0,) if variant=='S0' else (512,1024,2048);r={'id':ident,'root':Path(temporary)/'construction'/role,'method':variant,'checkpoints':{k:Path(temporary)/f'checkpoint{k}.pt' for k in ks}}
     for k,t,b in itertools.product(ks,(2,4),('raw','matched')):
      x.score(r,k,t,b,split='test');name,script,args,terminal=packets[-1];expected=out/'scores'/f'test_{ident}_k{k}_t{t}_{b}'
      assert Path(args[args.index('--out')+1])==expected and terminal==str(expected/'complete.json') and name==expected.name
   expected_count=48 if spec.get('execution_scope') else 156;assert len(packets)==expected_count;total+=len(packets)
  reports.append({'profile':folder.name,'score_packets':expected_count,'official_MTT_transfer_pairs':9,'status':'passed'})
 result={'status':'passed','profiles':reports,'real_score_packets_checked':total,'GPU_execution':False,'scope':'Configuration, source syntax, exact original dispatch names and official transfer tables; no model forward or new numerical GPU validation.'}
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
