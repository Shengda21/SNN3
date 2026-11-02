def require(ok,message):
 if not ok:raise RuntimeError(message)

def transported_values(g,b,gamma,beta,mu_source,var_source,mu_target,var_target,eps):
 """FP32 y_target→y_source affine; never divide by native gamma."""
 import torch
 require(all(x.dtype==torch.float32 for x in (g,b,gamma,beta,mu_source,var_source,mu_target,var_target)),'FP32 transport inputs')
 ss=torch.sqrt(var_source+eps);st=torch.sqrt(var_target+eps);a=st/ss;c=beta*(1.-a)+gamma*(mu_target-mu_source)/ss;u=1.+g
 return u*a-1.,b+u*c
