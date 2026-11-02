import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[1]))
"""Positive FP32 post-logit scale; unchanged full native mean forward underneath."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch import nn
from port_common import SCHEMA,VIEWS,require,sha
MODULE='_teacher_temperature'

class PositiveScales(nn.Module):
    def __init__(self,values,device):
        super().__init__();x=torch.as_tensor(values,dtype=torch.float32,device=device).clone()
        require(x.shape==(4,) and bool(torch.isfinite(x).all()) and bool((x>0).all()),'four positive finite FP32 beta')
        self.register_buffer('beta',x);self.reads=0;self.forbidden=False;self.calibration_bypasses=0;self.scaled_calls=0
    def for_view(self,t,matched):
        require(not self.forbidden,'calibration tried reading beta')
        require(t in (2,4) and isinstance(matched,bool),'actual native view')
        self.reads+=1;return self.beta[(0 if t==2 else 2)+int(matched)]
    @contextmanager
    def no_read(self):
        old=self.forbidden;before=self.reads;self.forbidden=True
        try:yield
        finally:
            self.forbidden=old;require(self.reads==before,'calibration beta getter read')

def attach(model,mean_bank,operator,values):
    require(MODULE not in model._modules,'temperature already attached')
    scale=PositiveScales(values,next(model.parameters()).device);model.add_module(MODULE,scale);original=model.forward
    def forward(self,*args,**kwargs):
        nodes=mean_bank._nodes(self);modes=[n.training for n in nodes.values()]
        require(all(modes) or not any(modes),'mixed BN collection modes')
        collecting=all(modes)
        z=original(*args,**kwargs)
        require(torch.is_tensor(z) and z.dtype==torch.float32,'complete original FP32 native mean logits')
        if collecting:
            scale.calibration_bypasses+=1;return z
        # Reproduce runtime.forward temporal readout before scaling, in FP32.
        if z.ndim==3:z=z.mean(0)
        require(z.ndim==2,'original native temporal readout must be 2D')
        z=z.float()
        # Actual bank predicate is the unchanged reviewed 12C operator predicate.
        matched=not operator._matches_raw(nodes,mean_bank)
        beta=scale.for_view(int(self.T),matched);out=z*beta
        require(out.dtype==torch.float32 and bool(torch.isfinite(out).all()),'finite FP32 scaled logits')
        scale.scaled_calls+=1;return out
    model.forward=MethodType(forward,model);return model

def load_deployment(root,architecture,checkpoint,window,runtime,native_loader=None):
    from package_common import parents
    tail,_=parents();ce=tail.ce_module();op=ce.feature_module();p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(p['schema']==SCHEMA and p['parent_schema']==tail.SCHEMA and p['architecture']==architecture and p['operator_source_sha256']==ce.OPERATOR_SHA,'own deployment/source/schema')
    require(p['temperature']['views']==list(VIEWS) and p['temperature']['rule']=='all48 fixed rho=0.5: beta_deploy=1+0.5*(all4_augmented_CE_beta-1); float32 cast once; no selection/refit','temperature contract')
    model=(native_loader or runtime.load_model)(root,architecture,checkpoint,window)
    bank=op.WindowBank(model);bank.load_state_dict(p['compensation'],strict=True)
    require(list(bank.names)==p['bn_names'] and list(bank.channels)==p['bn_channels'] and list(bank.eps)==p['bn_eps'],'original mean BN metadata')
    require(ce.equal_tree(bank.raw_mean,torch.cat([n.running_mean for n in bank._nodes(model).values()])) and ce.equal_tree(bank.raw_var,torch.cat([n.running_var for n in bank._nodes(model).values()])),'mean raw refs equal native S0')
    object.__setattr__(bank,'_feature_operator',op);op.attach(model,bank)
    import numpy as np
    require(p['temperature']['rho']==0.5 and p['temperature']['beta_float64_deploy']==[1.0+0.5*(b-1.0) for b in p['temperature']['beta_fit_all4']],'unchanged uniform half rule')
    values=np.asarray(p['temperature']['beta_float64_deploy'],dtype=np.float64).astype(np.float32)
    require(values.tolist()==p['temperature']['beta_float32'] and values.view(np.uint32).tolist()==p['temperature']['beta_float32_bits'],'original fit once-cast FP32 beta')
    return attach(model,bank,op,values)
