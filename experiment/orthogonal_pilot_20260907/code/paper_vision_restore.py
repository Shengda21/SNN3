"""Reconstruct the documented 30-epoch visual control if original assets are unavailable."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'direction1_budget_rotation_20260907';OUT=ROOT/'results/paper_experiments_20260908'
sys.path.insert(0,str(OLD/'code'))

def main(prepare_only=False):
    import numpy as np
    import torchvision
    data=OLD/'data';data.mkdir(exist_ok=True)
    ds=torchvision.datasets.CIFAR100(str(data),train=True,download=True)
    torchvision.datasets.CIFAR100(str(data),train=False,download=True)
    labels=np.asarray(ds.targets);rng=np.random.default_rng(20260907);tr=[];va=[]
    for label in np.unique(labels):
        indices=np.flatnonzero(labels==label);rng.shuffle(indices);n=len(indices)//10
        va.extend(indices[:n].tolist());tr.extend(indices[n:].tolist())
    (data/'CIFAR100_split.json').write_text(json.dumps({'train':sorted(tr),'validation':sorted(va),'seed':20260907}))
    prepared={'state':'ready','dataset':'CIFAR100','train_count':len(tr),'validation_count':len(va),
              'split_seed':20260907,'archive_md5':ds.tgz_md5,'source':ds.url,
              'integrity_checked_by':'torchvision CIFAR100','test_evaluated':False,'time':time.time()}
    (OUT/'E0').mkdir(parents=True,exist_ok=True)
    (OUT/'E0/visual_data.json').write_text(json.dumps(prepared,indent=2))
    if prepare_only:
        print(json.dumps(prepared),flush=True)
        return
    import train as v
    cfg=json.loads((OLD/'results/screen_identity_p0.0/config.json').read_text());cfg['id']='reconstructed_identity_p0.0'
    status={'state':'reconstructing','reason':'Original checkpoint unavailable locally and on the supplied empty cloud instance',
            'original_config':str(OLD/'results/screen_identity_p0.0/config.json'),'config':cfg,'exact_historical_weights':False,'started':time.time()}
    (OUT/'vision_status.json').write_text(json.dumps(status,indent=2))
    v.main(cfg)
    cp=OLD/'results'/cfg['id']/'best.pt';assert cp.exists()
    status.update(state='ready_reconstructed',checkpoint=str(cp),completed=time.time())
    (OUT/'vision_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    try:main(args.prepare_only)
    except Exception as e:
        p=OUT/('E0/visual_data.json' if args.prepare_only else 'vision_status.json');p.parent.mkdir(parents=True,exist_ok=True)
        state=json.loads(p.read_text()) if p.exists() else {}
        state.update(state='preparation_failed' if args.prepare_only else 'reconstruction_failed',error=str(e),time=time.time());p.write_text(json.dumps(state,indent=2));raise
