"""Summarize included route records, or independently recompute supplied prediction arrays."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(dataset,out,prediction_root=None,verify_recorded=False):
 root=HERE/'results'/dataset;rows=json.loads((root/'routes.json').read_bytes());assert len(rows)==312;groups={}
 for r in rows:
  if prediction_root is not None:
   candidates=[prediction_root/dataset/r['prediction'],prediction_root/r['prediction'],prediction_root/'scores'/r['id']/'scores.npz',prediction_root/r['id']/'scores.npz'];p=next((p for p in candidates if p.exists()),None);assert p is not None,'Missing predictions for '+r['id']
   if verify_recorded:assert digest(p)==r['prediction_sha256']
   with np.load(p,allow_pickle=False) as a:
    z,y,ids=(a[k] for k in ('logits','labels','example_id'));assert z.dtype==np.float32 and np.isfinite(z).all() and len(y)==r['samples'] and len(ids)==len(y);x=z.astype(np.float64);x-=x.max(1,keepdims=True);ce=float((np.log(np.exp(x).sum(1))-x[np.arange(len(y)),y]).mean());accuracy=float(np.mean(z.argmax(1)==y))
    if verify_recorded:assert abs(ce-r['ordinary_CE'])<1e-10 and accuracy==r['accuracy']
    r={**r,'ordinary_CE':ce,'accuracy':accuracy}
  key=tuple(r[k] for k in ('split','variant','K','window','BN'));groups.setdefault(key,[]).append(r)
 summaries=[]
 for k,g in sorted(groups.items()):
  assert len(g)==3 and {r['role'] for r in g}=={'formal1','formal2','formal3'};summaries.append(dict(zip(('split','variant','K','window','BN'),k),mean_CE=statistics.mean(r['ordinary_CE'] for r in g),sample_SD_CE=statistics.stdev(r['ordinary_CE'] for r in g),mean_accuracy=statistics.mean(r['accuracy'] for r in g),sample_SD_accuracy=statistics.stdev(r['accuracy'] for r in g)))
 out.mkdir(parents=True,exist_ok=True)
 with (out/(dataset+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summaries[0].keys());w.writeheader();w.writerows(summaries)
 g=json.loads((root/'gate.json').read_bytes());return {'dataset':dataset,'routes':len(rows),'scope':'supplied prediction metrics independently recomputed' if prediction_root is not None else 'included scalar route records summarized; no raw-array verification','recorded_formal_gate_passed':g.get('formal_gate_passed',g.get('gate_passed')),'status':'passed'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['all','cifar10','bloodmnist64','eurosat','plantwild'],default='all');p.add_argument('--output',type=Path,default=HERE.parent/'recomputed/extensions');p.add_argument('--prediction-root',type=Path);p.add_argument('--verify-recorded',action='store_true',help='Require exact recorded NPZ hashes and recorded metrics.');a=p.parse_args();assert not a.verify_recorded or a.prediction_root is not None;reports=[run(d,a.output,a.prediction_root,a.verify_recorded) for d in (['cifar10','bloodmnist64','eurosat','plantwild'] if a.dataset=='all' else [a.dataset])];print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
