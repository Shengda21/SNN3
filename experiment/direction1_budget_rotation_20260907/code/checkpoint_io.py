"""Load persistent model state without silently changing saved buffer precision."""
def load_preserving_buffers(model,state):
    changes=[]
    for name,buffer in list(model.named_buffers()):
        saved=state.get(name)
        if saved is not None and saved.dtype!=buffer.dtype:
            parent,field=name.rsplit('.',1) if '.' in name else ('',name)
            owner=model.get_submodule(parent) if parent else model
            changes.append({'name':name,'constructor_dtype':str(buffer.dtype),'checkpoint_dtype':str(saved.dtype)})
            setattr(owner,field,buffer.to(dtype=saved.dtype))
    model.load_state_dict(state,strict=True)
    for name,buffer in model.named_buffers():
        if name in state:
            assert buffer.dtype==state[name].dtype,(name,buffer.dtype,state[name].dtype)
    return changes
