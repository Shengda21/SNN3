"""Original CE byte-state checker, including signed zero and cross-device snapshots."""
import numpy as np

def equal_tree(left, right):

    """Exact tensor bytes, including dtype/shape and signed zero."""

    import torch

    if torch.is_tensor(left):

        return (torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape

                and left.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

                == right.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())

    if isinstance(left, np.ndarray):

        return (isinstance(right, np.ndarray) and left.dtype == right.dtype

                and left.shape == right.shape and left.tobytes() == right.tobytes())

    if isinstance(left, dict):

        return isinstance(right, dict) and set(left) == set(right) and all(equal_tree(v, right[k]) for k, v in left.items())

    if isinstance(left, (list, tuple)):

        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))

    return type(left) is type(right) and left == right
