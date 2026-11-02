import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[1]))
"""Fixed convex ordinary CE scalar rule. No validation or model dependencies."""
import numpy as np
BOX=(1/16,16.)
def evaluate(z,y,beta):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
    d=z-z.max(1,keepdims=True);s=beta*d;m=s.max(1,keepdims=True);e=np.exp(s-m);p=e/e.sum(1,keepdims=True)
    true=d[np.arange(len(y)),y];expected=(p*d).sum(1)
    return {'CE':float((m[:,0]+np.log(e.sum(1))-beta*true).mean()),'derivative':float((expected-true).mean()),'curvature':float((p*(d-expected[:,None])**2).sum(1).mean())}
def fit(z,y):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
    assert z.ndim==2 and y.shape==(len(z),) and np.isfinite(z).all() and len(z)>0
    lo,hi=BOX;lower=evaluate(z,y,lo);upper=evaluate(z,y,hi);flat=bool((z==z[:,:1]).all())
    for _ in range(80):
        mid=(lo+hi)/2
        if evaluate(z,y,mid)['derivative']>0:hi=mid
        else:lo=mid
    if flat:beta=1.;boundary='nonunique_flat'
    elif lower['derivative']>=0:beta=BOX[0];boundary='lower'
    elif upper['derivative']<=0:beta=BOX[1];boundary='upper'
    else:beta=(lo+hi)/2;boundary='interior'
    r=evaluate(z,y,beta);base=evaluate(z,y,1.)
    residual=max(0.,-r['derivative']) if boundary=='lower' else max(0.,r['derivative']) if boundary=='upper' else abs(r['derivative'])
    assert np.isfinite(beta) and beta>0 and r['CE']<=base['CE']+1e-10 and residual<1e-8
    pred=z.argmax(1);scaled32=np.multiply(z.astype(np.float32),np.float32(beta),dtype=np.float32)
    assert np.isfinite(scaled32).all()
    return {'beta':float(beta),'temperature':float(1/beta),'beta_box':list(BOX),'iterations':80,'boundary':boundary,'KKT_residual':float(residual),'at_beta1':base,'at_fitted_beta':r,'float64_argmax_flips':int(np.count_nonzero(pred!=(z*beta).argmax(1))),'cached_FP32_argmax_flips':int(np.count_nonzero(pred!=scaled32.argmax(1)))}
def analyze(z,y):
    formal=fit(z,y);a=fit(z[:4096],y[:4096]);b1=evaluate(z[4096:],y[4096:],1.);ba=evaluate(z[4096:],y[4096:],a['beta'])
    return {'fit_all4':formal,'diagnostic_Afit_Breport':{'role':'diagnostic_only_never_candidate','A_fit':a,'B_at_beta1':b1,'B_at_A_beta':ba,'B_CE_change':ba['CE']-b1['CE'],'B_never_selects_rule_or_beta':True}}
