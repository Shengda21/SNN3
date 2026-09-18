"""Exact torch fallback for upstream optional import; not a CUDA-kernel benchmark."""
def hadamard_transform(x, scale=1.0):
    import torch
    n=x.shape[-1]
    assert n>0 and n&(n-1)==0
    shape=x.shape; y=x.reshape(-1,n); h=1
    while h<n:
        z=y.reshape(-1,n//(2*h),2,h)
        y=torch.stack((z[:,:,0]+z[:,:,1],z[:,:,0]-z[:,:,1]),dim=2).reshape(-1,n)
        h*=2
    return y.reshape(shape)*scale
