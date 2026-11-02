"""API compatibility for the retained Spikformer source, never replacement dynamics.

The original clock_driven MultiStepLIFNode is preferred when available. With the
bundle's activation_based SpikingJelly release, use LIFNode(step_mode='m') with
the original tau, decay_input, threshold, reset, Sigmoid surrogate and detach_reset.
Full model/kernel parity remains a server validation item.
"""
BACKEND = 'cupy'

# SpikingJelly 0.0.0.0.14 CuPy type checks still use this removed NumPy alias.
# Match the original builtin-int identity; no numerical tensor path is changed.
import numpy as np
if 'int' not in np.__dict__:
    np.int = int


def MultiStepLIFNode(tau=2.,decay_input=True,v_threshold=1.,v_reset=0.,
                     surrogate_function=None,detach_reset=False,backend='cupy',**kwargs):
    selected = BACKEND
    try:
        from spikingjelly.clock_driven.neuron import MultiStepLIFNode as Original
        from spikingjelly.clock_driven import surrogate
        surrogate_function = surrogate.Sigmoid() if surrogate_function is None else surrogate_function
        return Original(tau=tau,decay_input=decay_input,v_threshold=v_threshold,v_reset=v_reset,
                        surrogate_function=surrogate_function,detach_reset=detach_reset,
                        backend=selected,**kwargs)
    except ImportError:
        from spikingjelly.activation_based import neuron,surrogate
        surrogate_function = surrogate.Sigmoid() if surrogate_function is None else surrogate_function
        return neuron.LIFNode(tau=tau,decay_input=decay_input,v_threshold=v_threshold,v_reset=v_reset,
                              surrogate_function=surrogate_function,detach_reset=detach_reset,
                              step_mode='m',backend=selected,**kwargs)


def reset_model(model):
    for module in model.modules():
        reset = getattr(module,'reset',None)
        if callable(reset):
            reset()
