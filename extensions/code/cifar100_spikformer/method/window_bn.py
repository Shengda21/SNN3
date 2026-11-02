"""Source-only S0 window/actual-bank-conditioned BN-output affine draft.

One native model and original head. This module never changes native parameters,
flags, pooling, neuron kernels or optimizer state. It is NOT the old T2 bypass.
"""
from contextlib import contextmanager
from types import MethodType
import torch
from torch import nn

SCHEMA='s0_window12_bn_output_draft_v1'
MODULE='_window12_bn'
ACTIVE='_window12_bn_context_active'
PARAMETER_NAMES=('t2_raw_g','t2_raw_b','t2_matched_g','t2_matched_b',
 't4_raw_early_g','t4_raw_early_b','t4_raw_late_g','t4_raw_late_b',
 't4_matched_early_g','t4_matched_early_b','t4_matched_late_g','t4_matched_late_b')

def require(ok,message):
 if not ok:raise RuntimeError(message)
def inventory(model):
 return {name:node for name,node in model.named_modules() if isinstance(node,(nn.BatchNorm1d,nn.BatchNorm2d))}
def _exact(left,right):
 return left.shape==right.shape and left.dtype==right.dtype and torch.equal(left,right.to(left.device))
def _matches_raw(nodes,bank):
 same=True
 for node,(begin,end) in zip(nodes.values(),bank.offsets):
  mean_equal=_exact(node.running_mean,bank.raw_mean[begin:end])
  var_equal=_exact(node.running_var,bank.raw_var[begin:end])
  same=same and mean_equal and var_equal
 return same

class WindowBank(nn.Module):
 def __init__(self,model,device=None):
  super().__init__();nodes=inventory(model)
  require(nodes and all(node.track_running_stats and node.running_mean is not None and node.running_var is not None for node in nodes.values()),'tracked native BN inventory')
  self.names=tuple(nodes);self.channels=tuple(int(node.num_features) for node in nodes.values());self.eps=tuple(float(node.eps) for node in nodes.values())
  total=0;offsets=[]
  for width in self.channels:offsets.append((total,total+width));total+=width
  self.offsets=tuple(offsets)
  if device is None:device=next(iter(nodes.values())).running_mean.device
  for name in PARAMETER_NAMES:setattr(self,name,nn.Parameter(torch.zeros(total,dtype=torch.float32,device=device)))
  self.register_buffer('raw_mean',torch.cat([node.running_mean.detach().to(device).clone() for node in nodes.values()]))
  self.register_buffer('raw_var',torch.cat([node.running_var.detach().to(device).clone() for node in nodes.values()]))
  require(self.raw_mean.dtype==self.raw_var.dtype==torch.float32,'S0 raw reference FP32')
  require(len({p.untyped_storage().data_ptr() for p in self.parameters()})==12,'twelve independent parameter-vector storages')
 def _nodes(self,model):
  nodes=inventory(model)
  require(tuple(nodes)==self.names and tuple(int(n.num_features) for n in nodes.values())==self.channels
   and tuple(float(n.eps) for n in nodes.values())==self.eps,'native inventory/epsilon changed')
  return nodes
 def parameter_names_for(self,window,matched):
  require(window in (2,4) and isinstance(matched,bool),'explicit native window and actual-bank boolean')
  if window==2:return PARAMETER_NAMES[2:4] if matched else PARAMETER_NAMES[:2]
  return PARAMETER_NAMES[8:12] if matched else PARAMETER_NAMES[4:8]
 def parameters_for(self,window,matched):return tuple(getattr(self,name) for name in self.parameter_names_for(window,matched))
 def parameters_from_counts(self,counts):
  require(counts['active_key'] is not None and counts['calibration_calls']==0,'an actual evaluated native view is required')
  return self.parameters_for(counts['window'],counts['matched'])
 @property
 def parameter_count(self):return sum(p.numel() for p in self.parameters())
 def metadata(self):
  return {'schema':SCHEMA,'BN_names':list(self.names),'BN_channels':list(self.channels),'BN_eps':list(self.eps),
   'parameter_names':list(PARAMETER_NAMES),'learned_scalars':self.parameter_count,'raw_reference_scalars':2*sum(self.channels),
   'groups':{str(t)+'_'+policy:list(self.parameter_names_for(t,policy=='matched')) for t in (2,4) for policy in ('raw','matched')}}

def require_distinct_matched_bank(model,bank,matched_bank):
 """Metadata preflight, outside calibration: actual statistics cannot label two groups if equal."""
 nodes=bank._nodes(model);same=True
 for node,name,(begin,end) in zip(nodes.values(),bank.names,bank.offsets):
  prefix=name+'.' if name else ''
  mean=matched_bank[prefix+'running_mean'];var=matched_bank[prefix+'running_var']
  require(mean.shape==node.running_mean.shape and var.shape==node.running_var.shape,'matched bank native schema')
  same=same and _exact(mean,bank.raw_mean[begin:end]) and _exact(var,bank.raw_var[begin:end])
 require(not same,'matched bank equals raw: actual-statistic predicate cannot select an independent matched adapter')
 return True

@contextmanager
def installed(model,bank,input_args=None):
 """Exactly one ORIGINAL native full-window call per context; no prefix or stepwise emulation.

 Bound-forward attachment primes with input_args since enclosing __call__
 prehooks already ran. BN collection exits before all map/reference VALUE reads.
 """
 window=int(model.T);require(window in (2,4),'only native T2/T4')
 require(not getattr(model,ACTIVE,False),'nonnested window12 context')
 nodes=bank._nodes(model);batch=[None];collecting=[None];matched=[None]
 counts={'window':window,'model_calls':0,'calls':0,'calibration_calls':0,'raw_calls':0,'matched_calls':0,
  'parameter_reads':0,'reference_reads':0,'active_key':None,'matched':None,
  'group_reads':{str(t)+'_'+p:0 for t in (2,4) for p in ('raw','matched')},
  'by_site':{name:{'calls':0,'calibration_calls':0} for name in nodes},'layouts':{}}
 had=ACTIVE in model.__dict__;old=model.__dict__.get(ACTIVE);handles=[]
 def root_before(_model,args):
  require(int(model.T)==window and len(args)==1 and torch.is_tensor(args[0]) and args[0].ndim==4,'native full-window image call')
  counts['model_calls']+=1;require(counts['model_calls']==1,'one native full forward per installed context')
  batch[0]=int(args[0].shape[0]);modes=[node.training for node in nodes.values()]
  require(all(modes) or not any(modes),'mixed native BN modes unsupported');collecting[0]=all(modes)
  if collecting[0]:return
  matched[0]=not _matches_raw(nodes,bank);counts['reference_reads']=2*len(nodes)
  counts['matched']=matched[0];counts['active_key']=str(window)+'_'+('matched' if matched[0] else 'raw')
 def make_hook(index,name):
  begin,end=bank.offsets[index];channels=bank.channels[index]
  def hook(node,args,y):
   require(batch[0] is not None and y.dtype==torch.float32 and y.ndim in (3,4)
    and y.shape[0]==window*batch[0] and y.shape[1]==channels,'native time-major [T*B,C,...]: '+name)
   counts['calls']+=1;counts['by_site'][name]['calls']+=1
   counts['layouts'][name]={'shape':list(y.shape),'time_major_leading_TB':True,'window':window,'batch':batch[0]}
   if node.training:
    require(collecting[0] is True,'collection mode changed within call')
    counts['calibration_calls']+=1;counts['by_site'][name]['calibration_calls']+=1;return None
   require(collecting[0] is False and isinstance(matched[0],bool),'actual-bank view not initialized')
   parameters=bank.parameters_for(window,matched[0]);shape=(1,channels)+(1,)*(y.ndim-2)
   if window==2:
    g,b=parameters;value=y*(1.+g[begin:end].reshape(shape))+b[begin:end].reshape(shape)
   else:
    eg,eb,lg,lb=parameters;boundary=2*batch[0]
    early=y[:boundary]*(1.+eg[begin:end].reshape(shape))+eb[begin:end].reshape(shape)
    late=y[boundary:]*(1.+lg[begin:end].reshape(shape))+lb[begin:end].reshape(shape)
    value=torch.cat((early,late),0)
   counts['parameter_reads']+=len(parameters);counts['group_reads'][counts['active_key']]+=len(parameters)
   counts['matched_calls' if matched[0] else 'raw_calls']+=1
   return value
  return hook
 object.__setattr__(model,ACTIVE,True)
 try:
  if input_args is None:handles.append(model.register_forward_pre_hook(root_before))
  for i,(name,node) in enumerate(nodes.items()):handles.append(node.register_forward_hook(make_hook(i,name)))
  if input_args is not None:root_before(model,input_args)
  yield counts
  require(counts['model_calls']==1 and all(x['calls']==1 for x in counts['by_site'].values()),'one full-window native forward with each BN once')
  if collecting[0]:require(counts['parameter_reads']==counts['reference_reads']==0 and counts['active_key'] is None,'all maps/reference values unread during calibration')
 finally:
  try:
   for handle in reversed(handles):handle.remove()
  finally:
   if had:object.__setattr__(model,ACTIVE,old)
   else:delattr(model,ACTIVE)

def attach(model,bank):
 require(MODULE not in model._modules,'window12 bank already attached')
 require(not hasattr(model,'_prefix_suffix_ce') and not hasattr(model,'_analytic_bn_state'),'native unwrapped model required')
 model.add_module(MODULE,bank);original=model.forward
 def forward(self,*args,**kwargs):
  require(int(self.T) in (2,4),'only native T2/T4')
  # Both windows use their own native complete forward; no old T2 early bypass.
  with installed(self,bank,input_args=args):return original(*args,**kwargs)
 model.forward=MethodType(forward,model);return model

def complete_view_gradient(loss,bank,counts):
 """Math-routing helper only; no optimizer. Loss must be the actual full native window LSCE.

 All active T4 four vectors receive the complete declared surrogate-graph loss
 derivative. There is no early-prefix/late-full split, PCGrad or inactive write.
 autograd.grad never populates native/head .grad; native flags are never toggled.
 """
 targets=bank.parameters_from_counts(counts)
 return torch.autograd.grad(loss,targets,allow_unused=False)

@contextmanager
def calibration_no_read_guard(bank):
 """Future actual-calibration check, not an executed test: getters and raw comparison are forbidden."""
 global _matches_raw
 old_match=_matches_raw;had='parameters_for' in bank.__dict__;old=bank.__dict__.get('parameters_for')
 def forbidden(*args,**kwargs):raise RuntimeError('calibration read window adapter/reference values')
 try:
  _matches_raw=forbidden;bank.parameters_for=forbidden;yield
 finally:
  try:_matches_raw=old_match
  finally:
   if had:bank.parameters_for=old
   else:delattr(bank,'parameters_for')
