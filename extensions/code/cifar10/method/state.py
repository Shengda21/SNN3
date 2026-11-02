

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def surrogate_nodes(model):
    return [(name, module) for name, module in model.named_modules()
            if any(cls.__name__ == 'BaseNode' and cls.__module__.startswith('spikingjelly.')
                   and '.neuron' in cls.__module__ for cls in type(module).__mro__)]


def clear_cache(model):
    if model._revision_architecture == 'resformer':
        model._revision_rv.v.clear_cache(model)


def dssa_flags(model):
    result = {}
    for name, module in model.named_modules():
        for key in ('init_firing_rate_x', 'init_firing_rate_attn'):
            if hasattr(module, key):
                value = getattr(module, key)
                require(type(value) is bool, 'Unexpected DSSA initialization flag type')
                result[(name, key)] = value
    return result


def persistent_buffers_names(model):
    return set(model.state_dict()) - set(dict(model.named_parameters()))


def set_locked_mode(model, rt):
    """Deploy-state normalization, differentiable neurons, no cached detach."""
    rt.reset(model)
    model.eval()
    clear_cache(model)
    nodes = surrogate_nodes(model)
    require(nodes, 'No recognized surrogate-gradient neuron nodes')
    for _, node in nodes:
        node.training = True
    node_ids = {id(node) for _, node in nodes}
    require(all(module.training == (id(module) in node_ids) for module in model.modules()),
            'A non-neuron entered training mode')


def state_snapshot(model):
    return {'state': {name: value.detach().clone() for name, value in model.state_dict().items()},
            'buffer_names': persistent_buffers_names(model), 'flags': dssa_flags(model),
            'modes': {name: module.training for name, module in model.named_modules()},
            'requires_grad': {name: p.requires_grad for name, p in model.named_parameters()},
            'T': int(model.T)}


def assert_state(model, saved):
    import torch
    current = model.state_dict()
    require(set(current) == set(saved['state']), 'Persistent state schema changed')
    require(all(current[name].dtype == value.dtype and torch.equal(current[name], value)
                for name, value in saved['state'].items()), 'Full model state or dtype changed')
    require(dssa_flags(model) == saved['flags'], 'DSSA flags not restored')
    require({name: m.training for name, m in model.named_modules()} == saved['modes'],
            'Module modes not restored')
    require({name: p.requires_grad for name, p in model.named_parameters()} == saved['requires_grad'],
            'Parameter trainability not restored')
    require(int(model.T) == saved['T'], 'Window not restored')


def restore_state(model, saved, rt):
    """Acceptance-only rollback restores registered buffers with their dtype."""
    import torch
    rt.reset(model)
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved['state'].items():
            if name in saved['buffer_names']:
                parent, _, leaf = name.rpartition('.')
                target = modules[parent]._buffers[leaf]
                modules[parent]._buffers[leaf] = value.to(device=target.device).clone()
            else:
                require(params[name].dtype == value.dtype, 'Acceptance changed parameter dtype')
                params[name].copy_(value)
    rt.set_window(model, saved['T'])
    clear_cache(model)
    for (name, key), value in saved['flags'].items():
        setattr(modules[name], key, value)
    for name, mode in saved['modes'].items():
        modules[name].training = mode
    for name, enabled in saved['requires_grad'].items():
        params[name].requires_grad_(enabled)
    model.zero_grad(set_to_none=True)
    assert_state(model, saved)
