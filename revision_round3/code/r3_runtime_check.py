"""Actual model forward/backward and inference checks on the new GPU stack."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
import cupy
from r2_vision import configure,load_model,WORK,v
from r2_score import training_inputs

configure()
info={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cupy':cupy.__version__,
      'nvrtc':list(cupy.cuda.nvrtc.getVersion()),'gpu':torch.cuda.get_device_name(),
      'capability':list(torch.cuda.get_device_capability()),'fp32':True,'tf32':False}
assert info['capability']==[12,0]
assert tuple(info['nvrtc'])>=(12,8)
xs,ys=training_inputs('vision')
rows=[]
for T in [2,4]:
    m,sites,_,_=load_model(T,training=True)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    x=xs[:64].cuda();y=ys[:64].cuda()
    tick=time.perf_counter()
    z=m(x).mean(0).float()
    loss=torch.nn.functional.cross_entropy(z,y)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    opt.step();v.functional.reset_net(m);v.clear_costs(sites)
    m.eval();v.prepare_inference(m)
    with torch.no_grad():
        a=m(x[:4]).mean(0).float();v.functional.reset_net(m);v.clear_costs(sites)
        b=m(x[:4]).mean(0).float();v.functional.reset_net(m);v.clear_costs(sites)
    assert torch.isfinite(loss) and torch.isfinite(a).all() and torch.equal(a,b)
    rows.append({'T':T,'loss':float(loss.detach()),'gradient_norm':float(norm),'reset_repeated_forward_equal':True,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.perf_counter()-tick})
    del m,sites,opt,x,y,z,loss,a,b
    import gc;gc.collect();torch.cuda.empty_cache()
info['checks']=rows;info['status']='passed'
out=WORK/'results/E0';out.mkdir(parents=True,exist_ok=True)
(out/'runtime_check.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info),flush=True)
