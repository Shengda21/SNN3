"""Original 8192 train-only augmented scalar fits, half scale and exact transport."""
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np
from port_common import *

def inputs(rt,root):
    import torch
    ids=np.random.default_rng(9640).choice(np.asarray(rt.split(root)['train']),2048,replace=False)
    require(not np.intersect1d(ids,rt.split(root)['validation']).size,'training-only fit')
    seeds=np.random.default_rng(9641).integers(0,2**32,8192,dtype=np.uint32)
    fullids=np.tile(ids,4);views=np.repeat(np.arange(4,dtype=np.int64),2048);tf=rt.training_transform();images=[]
    for i,s in zip(fullids,seeds):
        s=int(s);random.seed(s);np.random.seed(s);torch.random.default_generator.manual_seed(s);images.append(tf(rt.image(i)))
    images=torch.stack(images);labels=rt.raw()[1][fullids]
    cert={'samples':8192,'unique_training_ids':2048,'ids_sha256':hashlib.sha256(fullids.tobytes()).hexdigest(),
          'seeds_sha256':hashlib.sha256(seeds.tobytes()).hexdigest(),'images_sha256':hashlib.sha256(images.numpy().tobytes()).hexdigest(),
          'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),'test_read':False,'validation_read':False}
    return images,labels,fullids,seeds,views,cert

def run(args):
    import torch,ce,base
    from deploy import load,transport
    from ce_temperature import analyze
    rt=runtime(args.bundle,args.construction_root)
    from normalization import apply_bank
    root=Path(args.construction_root).resolve();train=Path(args.training).resolve();out=Path(args.out).resolve()
    require(not out.exists(),'preserve previous fit');out.mkdir(parents=True)
    terminal=read(train/'complete.json');require(terminal['status']=='complete' and terminal['attempts']==2048,'full trajectory barrier')
    guard_training();images,y,ids,seeds,views,cert=inputs(rt,root);write(out/'input_certificate.json',cert);records=[]
    for c in terminal['checkpoints']:
        require(sha(c['mean'])==c['mean_sha256'],'immutable mean');p=verify_checkpoint(root,c['mean'],'conditional')
        fits=[]
        for t in (2,4):
            for policy in ('raw','matched'):
                model=load(root,c['mean'],t,rt,scale=False);apply_bank(model,p['matched_banks'][str(t)] if policy=='matched' else base.persistent_bank(model))
                rt.prepare_mode(model,False);before=base.native_state(model,cpu=True);zs=[]
                with torch.no_grad():
                    for start in range(0,8192,64):zs.append(rt.forward(model,images[start:start+64].cuda()).cpu().numpy())
                z=np.concatenate(zs);require(z.shape==(8192,10) and z.dtype==np.float32 and np.isfinite(z).all(),'all augmented CIFAR10 logits')
                require(ce.equal_tree(before,base.native_state(model,cpu=True)),'native fit state unchanged')
                analysis=analyze(z,y);fits.append(analysis['fit_all4']['beta'])
                path=out/f'k{c["step"]}_t{t}_{policy}.npz';np.savez_compressed(path,logits=z,labels=y,example_id=ids,augmentation_seed=seeds,augmentation_view=views)
                r={'step':c['step'],'window':t,'BN':policy,'analysis':analysis,'arrays_sha256':sha(path)};records.append(r);write(path.with_suffix('.json'),r)
                del model
        effective=[1.+.5*(b-1.) for b in fits];beta=np.asarray(effective,dtype=np.float64).astype(np.float32)
        original=copy.deepcopy(p['compensation']);g,b=transport(p);p['compensation']['t2_matched_g']=g;p['compensation']['t2_matched_b']=b
        require(all(ce.equal_tree(v,p['compensation'][k]) for k,v in original.items() if k not in ('t2_matched_g','t2_matched_b')),'all other maps preserved')
        final=beta[[0,0,2,3]].copy();p.update(schema=SCHEMA,temperature_pending=False,
            temperature={'rho':.5,'views':list(VIEWS),'original_beta_fit_all4':fits,'original_effective_float64':effective,
                         'beta_float32':final.tolist(),'beta_float32_bits':final.view(np.uint32).tolist(),
                         'fit_source_views':['2_raw','2_raw','4_raw','4_matched'],'T2matched_refitted':False},
            covariant_revision={'arithmetic_device':'CPU','arithmetic_dtype':'torch.float32','bridge_sha256':sha(HERE/'bridge.py'),
                'previously_observed_formal_starts':False,'new_campaign':True,'source_mean_sha256':sha(c['mean'])})
        dest=out/f'final_step{c["step"]}.pt';rt.atomic_save(p,dest)
    write(out/'complete.json',{'status':'complete','schema':SCHEMA,'cells':len(records),'records':records,
          'source_training_complete_sha256':sha(train/'complete.json'),'test_read':False,'validation_read':False,
          'checkpoints':[{'step':k,'path':str(out/f'final_step{k}.pt'),'sha256':sha(out/f'final_step{k}.pt')} for k in BUDGETS]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--construction-root',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
