import numpy as np
ADAPTER_PREFIX="_window12_bn."

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def equal_tree(left, right):
    import torch

    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, np.generic):
        return (isinstance(right, np.generic) and left.dtype == right.dtype
                and bool(np.array_equal(left, right, equal_nan=True)))
    if isinstance(left, dict):
        return (isinstance(right, dict) and set(left) == set(right)
                and all(equal_tree(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(equal_tree(a, b) for a, b in zip(left, right)))
    return left == right


def native_state(model, cpu=False):
    result = {}
    for name, value in model.state_dict().items():
        if not name.startswith(ADAPTER_PREFIX):
            result[name] = value.detach().cpu().clone() if cpu else value.detach().clone()
    return result


def persistent_bank(model):
    from torch import nn

    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            prefix = name + "." if name else ""
            for key in ("running_mean", "running_var", "num_batches_tracked"):
                result[prefix + key] = getattr(module, key).detach().clone()
    require(result, "model has no BN bank")
    return result


def optimizer_steps(optimizer):
    values = []
    for state in optimizer.state.values():
        value = state.get("step", 0)
        values.append(int(value.item() if hasattr(value, "item") else value))
    return values


def cuda_batches(iterator):
    batches = []
    for _ in range(2):
        images, labels = next(iterator)
        require(len(labels) == 64, "microbatch size")
        batches.append((images.cuda(non_blocking=True),
                        labels.cuda(non_blocking=True)))
    return batches
