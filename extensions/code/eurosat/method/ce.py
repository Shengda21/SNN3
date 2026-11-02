import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[1]))
import copy,hashlib,json,math
from pathlib import Path
import base
import window_bn as operator
HERE=Path(__file__).resolve().parent
R=None
SCHEMA="eurosat64_full_window_ce_v1"
GROUPS=('t2_raw','t2_matched','t4_raw','t4_matched')
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
def base_module():return base
def feature_module():return operator


def require(ok,message):
 if not ok:raise RuntimeError(message)


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parameter_grads(model):return [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]


def cpu_tree(value):
 import torch
 if torch.is_tensor(value):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return type(value)(cpu_tree(x) for x in value)
 return copy.deepcopy(value)


def fingerprint(bank):
 h=hashlib.sha256()
 for n,v in sorted(bank.items()):
  x=v.detach().cpu().contiguous();h.update(json.dumps([n,list(x.shape),str(x.dtype)],separators=(',',':')).encode());h.update(x.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
 return h.hexdigest()


def group_info(key):return int(key[1]),key.endswith('matched')


def restore_optimizers(optimizers,states):
 errors=[]
 for key,optimizer in optimizers.items():
  try:optimizer.load_state_dict(states[key])
  except BaseException as exc:errors.append(exc)
 if errors:raise errors[0]


def build(job,rt,common=None):
 import torch
 if common is not None:common.verify_file(job['initial_checkpoint'],job['initial_checkpoint_sha256'])
 prefix=rt.load_model(R,job['architecture'],job['initial_checkpoint'],2);rt.prepare_mode(prefix,True)
 shadow=rt.load_model(R,job['architecture'],job['initial_checkpoint'],4)
 op=feature_module();bank=op.WindowBank(shadow);object.__setattr__(bank,'_feature_operator',op)
 require(len(bank.names)==job['bn_count'] and sum(bank.channels)==job['bn_channels_total'] and bank.parameter_count==job['adapter_parameter_count'],'native inventory and12C')
 optimizers={}
 for key in GROUPS:
  t,m=group_info(key);optimizers[key]=torch.optim.AdamW([{'params':bank.parameters_for(t,m),'param_names':list(bank.parameter_names_for(t,m))}],lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 return prefix,shadow,bank,optimizers


def prepare_banks(base,prefix,shadow,bank,worker,rt,apply_bank,force_exception=None):
 op=bank._feature_operator;saved=worker.state_snapshot(shadow);rng=copy.deepcopy(rt.rng_state());maps=cpu_tree(bank.state_dict());grads=parameter_grads(bank)
 banks={};receipts={};counts={};old_forward=rt.forward;committed=False
 def observed(model,images):
  if model is not shadow:return old_forward(model,images)
  with op.installed(shadow,bank) as c:value=old_forward(model,images)
  require(c['parameter_reads']==c['reference_reads']==0 and c['calibration_calls']==len(bank.names),'real calibration all maps/references unread')
  counts[str(int(model.T))]+=1;return value
 try:
  rt.forward=observed
  for t in (2,4):
   worker.restore_state(shadow,saved,rt);rt.set_window(shadow,t);counts[str(t)]=0
   with op.calibration_no_read_guard(bank):value,receipt=rt.calibrate(R,shadow,9640)
   require(counts[str(t)]==2048//rt.calibration_batch and receipt['samples']==2048 and receipt['batch']==rt.calibration_batch and receipt['calibration_seed']==9640,'one original calibration per window')
   op.require_distinct_matched_bank(shadow,bank,value)
   banks[str(t)]=cpu_tree(value);receipts[str(t)]=receipt
   if force_exception=='after_calibration_'+str(t):raise RuntimeError('intentional_'+force_exception)
  require(equal_tree(maps,bank.state_dict()) and equal_tree(grads,parameter_grads(bank)),'calibration map state unchanged')
  result={'banks':banks,'calibration':receipts,'fingerprints':{k:fingerprint(v) for k,v in banks.items()},'calibration_forward_calls':counts,'source_native_sha256':fingerprint(base.native_state(prefix,cpu=True))}
  worker.restore_state(shadow,saved,rt);rt.restore_rng(rng);committed=True
 finally:
  try:rt.forward=old_forward
  finally:
   try:worker.restore_state(shadow,saved,rt)
   finally:
    try:
     if not committed:
      bank.load_state_dict(maps,strict=True)
      for p,g in zip(bank.parameters(),grads):p.grad=g
    finally:rt.restore_rng(rng)
 return result


def phase(base,shadow,saved,bank,optimizer,key,batches,view_bank,worker,rt,apply_bank):
 import torch
 t,m=group_info(key);active=bank.parameters_for(t,m);inactive=tuple(p for p in bank.parameters() if all(p is not q for q in active))
 before=[p.detach().clone() for p in active];other=[p.detach().clone() for p in inactive];othergrads=[None if p.grad is None else p.grad.detach().clone() for p in inactive]
 grads=[torch.zeros_like(p) for p in active];losses=[];captures=[];op=bank._feature_operator
 for x,y in batches:
  worker.restore_state(shadow,saved,rt);apply_bank(shadow,view_bank);rt.set_window(shadow,t);worker.set_locked_mode(shadow,rt)
  with torch.enable_grad():
   with op.installed(shadow,bank) as counts:z=rt.forward(shadow,x)
   require(counts['active_key']==str(t)+'_'+('matched' if m else 'raw'),'actual bank selects declared group')
   loss=torch.nn.functional.cross_entropy(z,y,label_smoothing=0.0)
   derivative=op.complete_view_gradient(loss*(len(y)/128),bank,counts)
  require(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in derivative),'finite full-window CE/gradient')
  for a,b in zip(grads,derivative):a.add_(b.detach())
  require(all(p.grad is None for p in shadow.parameters()),'no native/head .grad writes')
  losses.append(float(loss.detach()));captures.append(counts)
  rt.reset(shadow);worker.clear_cache(shadow)
 worker.restore_state(shadow,saved,rt)
 optimizer.zero_grad(set_to_none=True)
 for p,g in zip(active,grads):p.grad=g
 norm=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True));require(math.isfinite(norm) and norm>=0,'finite active full-view gradient')
 optimizer.step()
 require(all(bool(torch.isfinite(p).all()) for p in active),'finite group parameters')
 changed=not equal_tree(before,[p.detach() for p in active])
 delta=math.sqrt(sum(float((p.detach().double()-old.double()).square().sum()) for p,old in zip(active,before)))
 require(equal_tree(other,[p.detach() for p in inactive]) and equal_tree(othergrads,[None if p.grad is None else p.grad.detach() for p in inactive]),'all inactive groups unmodified')
 return {'objective':sum(losses)/len(losses),'label_smoothing':0.0,'window':t,'actual_policy':'matched' if m else 'raw','gradient_norm':norm,'global_clip_norm':norm,
  'active_parameter_names':list(bank.parameter_names_for(t,m)),'optimizer_steps':sorted(set(base.optimizer_steps(optimizer))),
  'nonzero_update':changed,'parameter_delta_L2':delta,'complete_full_view_gradient':True,'inactive_groups_unchanged':True,'native_grad_none':True,'microbatch_captures':captures,'extra_head_calls':0}


def one_attempt(base,prefix,shadow,bank,optimizers,batches,worker,rt,apply_bank,banks,step,force_exception=None):
 require(len(batches)==128//rt.microbatch and all(len(y)==rt.microbatch for x,y in batches),'same original128 augmentation split2x64')
 entry=copy.deepcopy(rt.rng_state());native=cpu_tree(base.native_state(prefix,cpu=True));ps=worker.state_snapshot(prefix);ss=worker.state_snapshot(shadow)
 maps=cpu_tree(bank.state_dict());grad=parameter_grads(bank);states={k:copy.deepcopy(o.state_dict()) for k,o in optimizers.items()};bank_before=cpu_tree(banks)
 committed=False
 try:
  require(banks['source_native_sha256']==fingerprint(native),'stationary bank source is fixed native S0')
  raw=base.persistent_bank(prefix);result={}
  for key in GROUPS:
   t,m=group_info(key);bn=banks['banks'][str(t)] if m else raw
   result[key]=phase(base,shadow,ss,bank,optimizers[key],key,batches,bn,worker,rt,apply_bank)
   require(result[key]['optimizer_steps']==[step],'one active optimizer step per attempt')
   if force_exception=='after_'+key:raise RuntimeError('intentional_'+force_exception)
  require(equal_tree(bank_before,banks) and equal_tree(native,base.native_state(prefix,cpu=True)),'fixed native and cached bank bytes')
  result['transport']={'native_S0_unchanged':True,'native_grad_none':True,'rng_restored':True,'stationary_banks_unchanged':True,'full_gradient_all_active_vectors':True,'extra_head_calls':0}
  worker.restore_state(shadow,ss,rt);worker.restore_state(prefix,ps,rt);rt.restore_rng(entry)
  worker.assert_state(shadow,ss);worker.assert_state(prefix,ps);require(equal_tree(entry,rt.rng_state()),'RNG transaction')
  committed=True
 finally:
  if not committed:
   try:worker.restore_state(shadow,ss,rt)
   finally:
    try:worker.restore_state(prefix,ps,rt)
    finally:
     try:
      bank.load_state_dict(maps,strict=True)
      for p,g in zip(bank.parameters(),grad):p.grad=g
     finally:
      try:
       restore_optimizers(optimizers,states)
      finally:
       try:banks.clear();banks.update(bank_before)
       finally:rt.restore_rng(entry)
 return result


def pack(prefix,bank,initial,job,step,optimizers,banks,rng,identity):
 config=initial.get('initial_config',initial.get('config',{}))
 if 'initial_config' in config and 'dataset' not in config:config=config['initial_config']
 return {'schema':SCHEMA,'model':cpu_tree(base_module().native_state(prefix,cpu=True)),'compensation':cpu_tree(bank.state_dict()),
  'optimizers':{k:cpu_tree(v.state_dict()) for k,v in optimizers.items()},'matched_banks':cpu_tree(banks['banks']),
  'calibration':cpu_tree(banks['calibration']),'bank_fingerprints':banks['fingerprints'],'bank_source_native_sha256':banks['source_native_sha256'],
  'operator_source_sha256':sha(HERE/'window_bn.py'),'bn_names':list(bank.names),'bn_channels':list(bank.channels),'bn_eps':list(bank.eps),
  'adapter_parameter_count':bank.parameter_count,'architecture':job['architecture'],'initial_config':copy.deepcopy(config),
  'model_metadata':copy.deepcopy(prefix._revision_metadata),'identity':identity,'step':step,'logical_attempts':step,'rng':rng,
  'theta_optimizer':None,'theta_optimizer_updates':0,'group_optimizer_updates':{k:step for k in GROUPS},'frozen_S0_at_budget':True,
  'native_S0_checkpoint_sha256':job['initial_checkpoint_sha256'],'T2_identity_claim':False,'compute_matched':False,'formal_budget_eligible':True}
