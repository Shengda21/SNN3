"""Final native FP32 deployment with CPU covariant transport and half scaling."""
import numpy as np
import torch
from port_common import *
import ce

def load(root,checkpoint,window,rt,scale=True):
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(p['schema']==SCHEMA,'new campaign payload schema')
    require(p['operator_source_sha256']==sha(HERE/'window_bn.py'),'original feature operator source')
    if scale:
        gg,bb=transport(p)
        require(ce.equal_tree(gg,p['compensation']['t2_matched_g']) and ce.equal_tree(bb,p['compensation']['t2_matched_b']),'exact CPU coordinate revision')
        a=p['temperature'];fitted=a['original_beta_fit_all4']
        effective=[1.+.5*(b-1.) for b in fitted]
        values=np.asarray(effective,dtype=np.float64).astype(np.float32)[[0,0,2,3]].copy()
        require(a['original_effective_float64']==effective and a['beta_float32']==values.tolist()
                and a['beta_float32_bits']==values.view(np.uint32).tolist(),'half once-cast donor scale bytes')
    model=rt.load_model(root,'spikformer',checkpoint,window);op=ce.feature_module();bank=op.WindowBank(model)
    bank.load_state_dict(p['compensation'],strict=True)
    require(list(bank.names)==p['bn_names'] and list(bank.channels)==p['bn_channels'],'native inventory')
    object.__setattr__(bank,'_feature_operator',op);op.attach(model,bank)
    if scale:
        from temperature_wrapper import attach
        require(p['temperature_pending'] is False and p['temperature']['rho']==.5,'finished fixed half fit')
        model=attach(model,bank,op,np.asarray(p['temperature']['beta_float32'],dtype=np.float32))
    return model

def transport(p):
    from bridge import transported_values
    maps=p['compensation'];g,b=maps['t2_raw_g'],maps['t2_raw_b'];gg=[];bb=[];offset=0
    require(g.device.type==b.device.type=='cpu','CPU-only arithmetic')
    for name,width,eps in zip(p['bn_names'],p['bn_channels'],p['bn_eps']):
        prefix=name+'.' if name else '';lo,hi=offset,offset+width;offset=hi
        source=p['model'];target=p['matched_banks']['2']
        a,c=transported_values(g[lo:hi],b[lo:hi],source[prefix+'weight'],source[prefix+'bias'],
             source[prefix+'running_mean'],source[prefix+'running_var'],target[prefix+'running_mean'],target[prefix+'running_var'],eps)
        gg.append(a);bb.append(c)
    require(offset==g.numel(),'complete native BN transport')
    return torch.cat(gg),torch.cat(bb)
