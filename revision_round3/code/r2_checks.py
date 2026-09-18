"""Training-side migration checks, without loading either official test set."""
import json,subprocess,sys,time
import numpy as np
import torch
from torch import nn
from torchvision import datasets
from r2_vision import *


@torch.no_grad()
def calibrate_bn(m,sites,images):
    m.eval().requires_grad_(False);v.prepare_inference(m)
    before=state_cpu(m);bn_names=set();modules=[]
    for name,module in m.named_modules():
        if isinstance(module,nn.BatchNorm2d):
            modules.append((module,module.momentum))
            bn_names.update(name+'.'+k for k in ['running_mean','running_var','num_batches_tracked'])
            module.reset_running_stats();module.momentum=None;module.train()
    assert len(modules)==33
    for i in range(0,len(images),64):
        v.functional.reset_net(m);v.clear_costs(sites)
        z=m(images[i:i+64].cuda()).mean(0)
        assert torch.isfinite(z).all()
    v.functional.reset_net(m);v.clear_costs(sites)
    for module,momentum in modules:module.eval();module.momentum=momentum
    after=state_cpu(m)
    protected=[k for k in before if k not in bn_names]
    assert all(before[k].dtype==after[k].dtype and torch.equal(before[k],after[k]) for k in protected)
    return {k:after[k] for k in bn_names},{'bn_layers':len(modules),'protected_state_entries':len(protected),
       'protected_values_and_dtypes_unchanged':True,'samples':len(images),'batches':len(images)//64}


def calibration_inputs():
    split=json.loads((VISION/'data/CIFAR100_split.json').read_text())
    ids=np.random.default_rng(9640).choice(np.array(split['train']),2048,replace=False)
    base=datasets.CIFAR100(str(VISION/'data'),train=True,transform=evaluate_transform())
    return torch.stack([base[int(i)][0] for i in ids]),ids


def resume_check():
    base=[sys.executable,str(Path(__file__).with_name('r2_vision.py')),'--T','4','--seed','9699','--lr','2e-5',
          '--tag','resumed','--stage','probe']
    subprocess.run(base+['--steps','2'],check=True)
    subprocess.run(base+['--steps','4'],check=True)
    a=torch.load(WORK/'models/probe/continuity/resume.pt',map_location='cpu',weights_only=False)
    b=torch.load(WORK/'models/probe/resumed/resume.pt',map_location='cpu',weights_only=False)
    eq=all(torch.equal(a['model'][k],b['model'][k]) and a['model'][k].dtype==b['model'][k].dtype for k in a['model'])
    opt=all(torch.equal(x,b['optimizer']['state'][k][n]) if torch.is_tensor(x) else x==b['optimizer']['state'][k][n]
         for k,state in a['optimizer']['state'].items() for n,x in state.items())
    ds0=PlannedSamples(9699,4);ds1=PlannedSamples(9699,4,start=2)
    items=[256,300,511]
    samples=all(torch.equal(ds0[i][0],ds1[i-256][0]) for i in items)
    result={'weights_and_buffers_bitwise_equal':eq,'optimizer_bitwise_equal':opt,'resumed_augmentation_samples_equal':samples}
    write_json(WORK/'results/E0/resume_check.json',result)
    assert all(result.values()),result


@torch.no_grad()
def vision_check():
    configure();images,ids=calibration_inputs();checks=[]
    np.save(WORK/'results/E0/bn_train_ids.npy',ids)
    for T in [2,4]:
        m,sites,_,_=load_model(T)
        x=images[:4].cuda().clone()
        def call():
            v.functional.reset_net(m);v.clear_costs(sites);return m(x).mean(0).float()
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):z=call()
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):z=call()
        aba=[]
        for start in [0,64,0]:
            x.copy_(images[start:start+4]);expected=call();graph.replay()
            aba.append({'bitwise_equal':torch.equal(expected,z),'max_error':float((expected-z).abs().max())})
        assert all(r['bitwise_equal'] for r in aba)
        del graph,z,x,expected
        delta,cal=calibrate_bn(m,sites,images)
        checks.append({'T':T,'graph_ABA':aba,'calibration':cal,'firing_buffer_dtypes':
            {k:str(x.dtype) for k,x in m.named_buffers() if 'firing_rate' in k}})
        del m,sites,delta;gc.collect();torch.cuda.empty_cache()
    write_json(WORK/'results/E0/vision_migration.json',checks)


if __name__=='__main__':
    resume_check();vision_check();print('VISION_CHECKS_PASSED',flush=True)
