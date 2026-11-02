"""Official CIFAR Spikformer model transplanted to the 100-class study task.

The source and compatible derivative are both retained under vendor/spikformer.
This is not an official CIFAR-100 training configuration or pretrained model.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

BUNDLE = Path(__file__).resolve().parents[1]
VENDOR = BUNDLE/'vendor/spikformer'
_MODULE = None


def _load_source(path,name='snn_revision2_spikformer'):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    # timm.register_model resolves fn.__module__ through sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name,None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _registry_import_self_check():
    """Exercise timm-like registration and failed imports without model dependencies."""
    from unittest.mock import patch
    class RegistryLoader:
        def create_module(self,spec):
            return None
        def exec_module(self,module):
            source = ('import sys\ndef register(fn):\n'
                ' assert sys.modules[fn.__module__] is sys.modules[__name__]\n'
                ' return fn\n@register\ndef fixture(): return 7\n')
            exec(source,module.__dict__)
    class FailureLoader(RegistryLoader):
        def exec_module(self,module):
            super().exec_module(module)
            raise RuntimeError('software fixture expected failure')
    name = 'snn_revision2_registry_software_fixture'
    previous = sys.modules.get(name)
    cached = _MODULE
    try:
        with patch.object(importlib.util,'spec_from_file_location',
                side_effect=lambda n,p:importlib.util.spec_from_loader(n,RegistryLoader())):
            loaded = _load_source('unused',name)
            assert loaded.fixture()==7 and sys.modules[name] is loaded
        sys.modules.pop(name)
        with patch.object(importlib.util,'spec_from_file_location',
                side_effect=lambda n,p:importlib.util.spec_from_loader(n,FailureLoader())):
            try:
                _load_source('unused',name)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Failed module initialization was accepted')
        assert name not in sys.modules and _MODULE is cached
    finally:
        if previous is None:
            sys.modules.pop(name,None)
        else:
            sys.modules[name] = previous
    return {'decorator_registry_lookup':'passed','failed_import_cleaned':'passed',
            'half_initialized_module_not_cached':True,'software_fixture_only':True}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_window(model,T):
    if T not in (2,4):
        raise ValueError('This protocol defines only T=2 and T=4')
    model.T = int(T)


def build_model(architecture,T,seed,device='cuda'):
    if architecture.lower() not in ('spikformer','spikformer-4-384','spikformer4_384'):
        raise ValueError('backbone.py owns the Spikformer branch; resformer is supplied by runtime.py')
    import numpy as np
    import torch
    from torch import nn
    import spikformer_compat as compatibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    global _MODULE
    if _MODULE is None:
        # Cache only a successfully initialized module.
        _MODULE = _load_source(VENDOR/'model_compatible.py')
    old_backend = compatibility.BACKEND
    compatibility.BACKEND = 'cupy' if str(device).startswith('cuda') else 'torch'
    try:
        model = _MODULE.Spikformer(img_size_h=32,img_size_w=32,patch_size=4,
            in_channels=3,num_classes=100,embed_dims=384,num_heads=12,
            mlp_ratios=4,qkv_bias=False,depths=4,sr_ratios=1,T=int(T)).to(device)
    finally:
        compatibility.BACKEND = old_backend
    set_window(model,T)
    def reset_fn():
        compatibility.reset_model(model)
    bn = [(name,m) for name,m in model.named_modules() if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d))]
    bn_keys = {name+'.'+key for name,_ in bn for key in ('running_mean','running_var','num_batches_tracked')}
    parameter_names = set(dict(model.named_parameters()))
    persistent = sorted(set(model.state_dict())-parameter_names)
    non_bn = sorted(set(persistent)-bn_keys)
    metadata = {'architecture':'Spikformer-4-384','source_repository':'https://github.com/ZK-Zhou/spikformer',
        'source_commit':'621268981827af47aa666c1b5f774de4281d90ad',
        'source_model':'cifar10/model.py','classification_transfer':'official CIFAR-10 model definition with num_classes=100',
        'T':int(T),'input_size':[3,32,32],'depth':4,'embed_dim':384,'heads':12,
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'bn1d_layers':sum(isinstance(m,nn.BatchNorm1d) for _,m in bn),
        'bn2d_layers':sum(isinstance(m,nn.BatchNorm2d) for _,m in bn),
        'persistent_nonparameter_keys':persistent,'non_BN_persistent_keys':non_bn,
        'only_BN_persistent_state_verified_at_construction':not non_bn,
        'model_original_sha256':_hash(VENDOR/'model_original.py'),
        'model_compatible_sha256':_hash(VENDOR/'model_compatible.py'),
        'return_shape':'batch by class; temporal averaging occurs in official forward',
        'server_full_model_forward_backward_validation':'pending'}
    return model,reset_fn,metadata


def self_check():
    original = (VENDOR/'model_original.py').read_text(encoding='utf-8')
    compatible = (VENDOR/'model_compatible.py').read_text(encoding='utf-8')
    old = 'from spikingjelly.clock_driven.neuron import MultiStepLIFNode'
    new = 'from spikformer_compat import MultiStepLIFNode'
    assert original.count(old)==1 and compatible==original.replace(old,new)
    tree = ast.parse(original)
    classes = {node.name for node in tree.body if isinstance(node,ast.ClassDef)}
    assert {'MLP','SSA','Block','SPS','Spikformer'}.issubset(classes)
    assert 'self.T = T' in original and 'self.head(x.mean(0))' in original
    expected = {'model_original.py':'c87bc7afbc70f331781b99be58483a813d5b657c',
        'cifar10_original.yml':'c22855f72d1bc1dc58496779ef5235dddccf192a',
        'LICENSE':'d135e22004c63a015849d5fa059191aa7a9a23a8'}
    for name,identity in expected.items():
        data = (VENDOR/name).read_bytes()
        observed = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert observed==identity, f'Original upstream blob changed: {name}'
    return {'status':'passed','scope':'source AST and single import substitution only',
        'official_original_retained':True,'compatible_change_is_only_import':True,
        'three_original_files_match_official_git_blob_SHA1':True,
        'dynamic_import_registry_lifecycle':_registry_import_self_check(),
        'full_model_CPU_test':'not executed; local torchvision/timm/spikingjelly are absent',
        'full_model_GPU_test':'pending server','GPU_experiments_executed':False}


if __name__=='__main__':
    print(json.dumps(self_check(),indent=2))
