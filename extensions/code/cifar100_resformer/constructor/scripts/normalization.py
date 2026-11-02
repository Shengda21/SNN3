"""BN-only calibration for BatchNorm1d/2d; no neuron or parameter updates.

On success the newly estimated BN running state remains installed. Every module's
training flag and every parameter's requires_grad flag is restored. Transient
neuron state is reset before each call and on exit. A failed calibration restores
the original persistent state before propagating the error.
"""
from __future__ import annotations
import torch
from torch import nn


def _cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def _bn_modules_and_keys(model):
    modules, keys = [], set()
    for name,module in model.named_modules():
        if isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):
            if not module.track_running_stats:
                raise ValueError(f'BN without persistent statistics is unsupported: {name}')
            modules.append((name,module))
            prefix = name+'.' if name else ''
            keys.update(prefix+k for k in ('running_mean','running_var','num_batches_tracked'))
    if not modules:
        raise ValueError('No BatchNorm1d/2d layers found')
    return modules,keys


def _unchanged(before,after,keys):
    return all(before[k].dtype==after[k].dtype and torch.equal(before[k],after[k]) for k in keys)


@torch.no_grad()
def calibrate_bn(model,images,reset_fn,forward_fn,batch=64):
    if batch <= 0 or len(images)==0:
        raise ValueError('Calibration requires nonempty images and a positive batch')
    modules,bn_keys = _bn_modules_and_keys(model)
    before = _cpu_state(model)
    protected = set(before)-bn_keys
    modes = [(module,module.training) for module in model.modules()]
    gradients = [(parameter,parameter.requires_grad) for parameter in model.parameters()]
    momenta = [(module,module.momentum) for _,module in modules]
    device = next(model.parameters()).device
    success = False
    try:
        model.eval()
        for parameter,_ in gradients:
            parameter.requires_grad_(False)
        for _,module in modules:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        batches = 0
        for start in range(0,len(images),batch):
            reset_fn()
            output = forward_fn(images[start:start+batch].to(device))
            if not torch.is_tensor(output) or not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite or unsupported calibration output')
            batches += 1
        reset_fn()
        after = _cpu_state(model)
        if not _unchanged(before,after,protected):
            raise AssertionError('Calibration modified protected parameters or non-BN buffers')
        bank = {name:after[name] for name in sorted(bn_keys)}
        report = {'bn_layers':len(modules),
            'bn1d_layers':sum(isinstance(m,nn.BatchNorm1d) for _,m in modules),
            'bn2d_layers':sum(isinstance(m,nn.BatchNorm2d) for _,m in modules),
            'samples':len(images),'batches':batches,'batch':batch,
            'protected_state_entries':len(protected),
            'protected_values_and_dtypes_unchanged':True,
            'parameter_flags_and_module_modes_restored':True,
            'calibration_changes_only_BN_running_state':True}
        success = True
        return bank,report
    finally:
        if not success:
            current = model.state_dict()
            for name,value in before.items():
                current[name].copy_(value.to(current[name].device))
        for module,momentum in momenta:
            module.momentum = momentum
        # Assign flags individually; recursive train() would overwrite child modes.
        for module,mode in modes:
            module.training = mode
        for parameter,flag in gradients:
            parameter.requires_grad_(flag)
        reset_fn()


@torch.no_grad()
def apply_bank(model,bn_delta):
    modules,keys = _bn_modules_and_keys(model)
    if set(bn_delta)!=keys:
        raise ValueError('BN bank keys do not exactly match this model')
    before = _cpu_state(model)
    current = model.state_dict()
    for name,value in bn_delta.items():
        if value.shape!=current[name].shape or value.dtype!=current[name].dtype:
            raise ValueError(f'BN bank tensor identity mismatch: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite BN bank: {name}')
    for name,value in bn_delta.items():
        current[name].copy_(value.to(current[name].device))
    after = _cpu_state(model)
    assert _unchanged(before,after,set(before)-keys)
    return {'bn_layers':len(modules),'bank_entries':len(keys),
            'protected_values_and_dtypes_unchanged':True,'loaded_saved_statistics':True}


def self_check():
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3,4,1)
            self.bn2 = nn.BatchNorm2d(4)
            self.bn1 = nn.BatchNorm1d(4)
            self.drop = nn.Dropout(.5)
            self.register_buffer('protected_counter',torch.tensor(7,dtype=torch.int64))
        def forward(self,x):
            return self.bn1(self.bn2(self.conv(x)).mean((2,3)))
    torch.manual_seed(20261020)
    model = Fixture().train()
    model.bn1.eval()
    next(model.parameters()).requires_grad_(False)
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    before = _cpu_state(model)
    bank,report = calibrate_bn(model,torch.randn(8,3,4,4),lambda:None,model,batch=4)
    assert report['bn1d_layers']==report['bn2d_layers']==1
    assert [m.training for m in model.modules()]==modes
    assert [p.requires_grad for p in model.parameters()]==flags
    assert int(bank['bn1.num_batches_tracked'])==int(bank['bn2.num_batches_tracked'])==2
    assert not torch.equal(before['bn2.running_mean'],bank['bn2.running_mean'])
    current = _cpu_state(model)
    apply_bank(model,bank)
    assert _unchanged(current,_cpu_state(model),current)
    try:
        calibrate_bn(model,torch.randn(8,3,4,4),lambda:None,
                     lambda x:model(x)*float('nan'),batch=4)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Nonfinite calibration was not rejected')
    assert _unchanged(current,_cpu_state(model),current)
    assert [m.training for m in model.modules()]==modes
    assert [p.requires_grad for p in model.parameters()]==flags
    return {'status':'passed','software_fixture_only':True,'BN1d_and_BN2d':True,
        'protected_parameters_and_buffers':True,'mode_and_gradient_flags_restored':True,
        'failure_rolls_back_persistent_state':True,'GPU_experiments_executed':False}


if __name__=='__main__':
    import json
    print(json.dumps(self_check(),indent=2))
