"""Unique changed-path acceptance: CIFAR10, AMP constructor and held-out validation."""
import argparse
from pathlib import Path
import sys
import json
import construct

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--data',type=Path,required=True);args=a.parse_args()
    root=construct.prepare(args.root,args.data)
    import torch
    import common,runtime as rt
    p=construct.read(construct.B/'protocol.json');common.protocol=lambda:p;rt.protocol=lambda:p
    rt.seed_all(22650)
    model=rt.build_model(root,'resformer',4,22650)
    loader,_=rt.construction_loaders(root,'C-conf',0)
    x,y=next(iter(loader));assert len(y)==128 and int(y.max())<10
    from timm.data.mixup import Mixup
    mix=Mixup(mixup_alpha=.8,cutmix_alpha=1.,prob=1.,switch_prob=.5,mode='batch',label_smoothing=.1,num_classes=10)
    rt.prepare_mode(model,True);x,target=mix(x.cuda(),y.cuda())
    with torch.autocast('cuda',dtype=torch.float16):
        z=rt.forward(model,x);loss=-(target*torch.nn.functional.log_softmax(z,-1)).sum(-1).mean()
    assert z.shape==(128,10) and bool(torch.isfinite(loss))
    loss.backward();assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
    model.zero_grad(set_to_none=True);rt.reset(model)
    validation=rt.evaluate_validation(root,model)
    assert validation['samples']==5000 and 0<=validation['accuracy']<=1
    construct.write(root/'environment/cifar10_acceptance.json',{'status':'passed','AMP_output_shape':[128,10],
                    'finite_loss_and_gradients':True,'validation':validation,'optimizer_steps':0,
                    'test_read':False,'publication_result':False})
    print(json.dumps({'status':'passed','validation':validation}))

if __name__=='__main__':main()
