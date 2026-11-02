"""Prepare original train/fit packets for independently measured local execution."""
import argparse,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--role',choices=['formal1','formal2','formal3'],required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.work=a.work.resolve()
 sys.path.insert(0,str(ROOT));import run
 c,B,spec,protocol,out,gpu=run.setup(a);run.fixed_inputs(c,B,spec,out);original=run.load(B/'method/controller.py','resource_packet_producer');stages=[]
 class Capture(original.Campaign):
  def stage(self,name,script,args,terminal,wait=True):stages.append({'id':name,'argv':[sys.executable,str(ROOT/'entry.py'),'--config',str(a.work/'run.json'),'--script',script,'--',*map(str,args)],'cwd':str(ROOT),'timeout_seconds':172800,'terminal_files':[str(terminal)],'must_not_exist':[str(terminal)]})
 x=Capture({'output':str(out),'constructor_bundle':str(B/'constructor'),'construction_parent':c['construction_parent']});x.conditional(a.role,spec['formal_seeds'][a.role]);assert len(stages)==2
 a.output.write_text(json.dumps({'python':sys.executable,'physical_gpu':gpu,'selection_uses_test':False,'historical_energy_reconstruction':False,'sample_interval_seconds':0.5,'stages':stages},indent=2));print('CPU plan prepared; no GPU execution')
if __name__=='__main__':main()
