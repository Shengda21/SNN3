import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[2]))
"""Stage-wise temporal configurations on the unchanged SpikingResformer.

The official table_spwn function is loaded from the bundled MIT-licensed
reference. No approximate interpolation replaces its sum/repeat operation.
"""
import ast
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=9)
def official_table(in_T, out_T):
    source=Path(__file__).resolve().parents[1]/'upstream/temporal_flexibility_in_SNN-main/train-time-stepped/utils.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='table_spwn')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['table_spwn'](in_T,out_T)


def temporal_transfer(x, out_T):
    import torch
    if x.shape[0]==out_T:return x
    groups=official_table(x.shape[0],out_T)
    return torch.stack([x[indices].sum(dim=0) for indices in groups],dim=0)


class StageAdapter:
    def __init__(self, model):
        if len(model.layers)!=3:raise ValueError('This adapter is defined for the three retained stages')
        self.model=model;self.pattern=None;self.handles=[]
        for i,stage in enumerate(model.layers):
            def before(module, inputs, index=i):
                return inputs if self.pattern is None else (temporal_transfer(inputs[0],self.pattern[index]),)
            self.handles.append(stage.register_forward_pre_hook(before))

    def set(self, pattern):
        self.pattern=None if pattern is None else tuple(map(int,pattern))
        if self.pattern is not None:
            if len(self.pattern)!=3 or any(t<2 or t>4 for t in self.pattern):raise ValueError(self.pattern)
            self.model.T=self.pattern[0]

    def close(self):
        for handle in self.handles:handle.remove()


def local_table_check():
    # Covers precisely the temporal values used by the configured method.
    for i in (2,3,4):
        for o in (2,3,4):
            groups=official_table(i,o)
            assert len(groups)==o and all(groups)
            assert all(0<=k<i for group in groups for k in group)
    assert official_table(4,2)==[[0,1],[2,3]]
    assert official_table(2,4)==[[0],[0],[1],[1]]
    return {'pairs':9,'downsample':'sum','upsample':'repeat','official_source_used':True}
