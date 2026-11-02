"""Native batch64 FP32 scoring; every row retains actual hardware and input IDs."""
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
from port_common import *

def run(args):
    import torch,base,ce
    rt=runtime(args.bundle,args.construction_root)
    from normalization import apply_bank
    checkpoint=Path(args.checkpoint).resolve();out=Path(args.out).resolve();root=Path(args.construction_root).resolve()
    require(not out.exists(),'preserve score artifacts');out.mkdir(parents=True)
    if args.split=='test':
        verify_test_freeze(args.test_freeze,checkpoint,root,args.method)
    p=verify_checkpoint(root,checkpoint,args.method)
    variant=args.method
    if args.method=='native':variant='mtt' if p['config']['method']=='mtt' else 'native'+str(p['config']['T'])
    if args.method=='conditional':
        from deploy import load
        model=load(root,checkpoint,args.window,rt)
        require(p['temperature_pending'] is False,'final full method only')
    else:model=rt.load_model(root,'resformer',checkpoint,args.window)
    if args.policy=='matched':
        if args.method=='conditional':
            bank=p['matched_banks'][str(args.window)];receipt=p['calibration'][str(args.window)]
        else:bank,receipt=rt.calibrate(root,model,9640)
        apply_bank(model,bank)
    else:bank=base.persistent_bank(model);receipt={'policy':'native_raw','calibrations':0}
    rt.prepare_mode(model,False);before=base.native_state(model,cpu=True)
    if args.split=='validation':ids=np.asarray(rt.split(root)['validation'],dtype=np.int64);raw,labels=rt.raw()
    else:
        with (rt.vision_root(root)/'data/cifar-100-python/test').open('rb') as f:d=pickle.load(f,encoding='latin1')
        raw,labels=d['data'],np.asarray(d['fine_labels'],dtype=np.int64);ids=np.arange(10000,dtype=np.int64)
    from PIL import Image
    tf=rt.evaluate_transform();zs=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            selected=ids[start:start+64];x=torch.stack([tf(Image.fromarray(raw[int(i)].reshape(3,32,32).transpose(1,2,0))) for i in selected])
            zs.append(rt.forward(model,x.cuda()).cpu().numpy())
    z=np.concatenate(zs);y=labels[ids]
    require(z.shape==(len(ids),100) and z.dtype==np.float32 and np.isfinite(z).all(),'complete native logits')
    require(ce.equal_tree(before,base.native_state(model,cpu=True)),'persistent native state unchanged')
    x=z.astype(np.float64);x-=x.max(1,keepdims=True);loss=np.log(np.exp(x).sum(1))-x[np.arange(len(y)),y]
    np.savez_compressed(out/'scores.npz',logits=z,labels=y,example_id=ids)
    write(out/'complete.json',{'status':'complete','schema':SCHEMA,'method':args.method,'window':args.window,'BN':args.policy,
          'split':args.split,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
          'samples':len(y),'ordinary_CE':float(loss.mean()),'accuracy':float(np.mean(z.argmax(1)==y)),
          'example_identity_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
          'label_identity_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
          'start_sha256':read(root/'construction/C-conf/complete.json')['checkpoint_sha256'],
          'authority':verify_parent(args.bundle,root),'checkpoint_step':p.get('step',0),
          'train_T':p.get('config',{}).get('T') if args.method=='native' else None,
          'adaptation_seed':(p['config']['seed'] if args.method=='native' else p['identity']['job']['seed'] if args.method=='conditional' else None),
          'variant':variant,
          'bank_fingerprint':ce.fingerprint(bank),'calibration':receipt,'arrays_sha256':sha(out/'scores.npz'),
          'hardware':read(root/'environment/actual_hardware.json'),'FP32':True,'TF32':False,'batch':64})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--construction-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--method',choices=['conditional','native','S0'],required=True)
    p.add_argument('--window',type=int,choices=[2,4],required=True);p.add_argument('--policy',choices=['raw','matched'],required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--test-freeze',type=Path)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
