"""Full-model GWFFN output spike coding; no backbone width/depth reduction.

Column-vector convention: s=SN(Qx), y=(W Q.T)s. Only the decoder is
folded. The online encoder is retained and included in operation accounting.
"""
import math,weakref
import torch
from torch import nn
from torch.nn import functional as F

class ChannelRotation(nn.Module):
    def __init__(self,channels,group,kind):
        super().__init__()
        assert channels%group==0 and group&(group-1)==0
        self.channels=channels;self.group=group;self.kind=kind
        self.groups=channels//group;self.stages=int(math.log2(group))
        h=torch.ones(1,1)
        for _ in range(self.stages): h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)/math.sqrt(2)
        self.register_buffer('had',h.expand(self.groups,-1,-1).clone())
        if kind=='givens':
            # Identity initialization, so the intervention starts at the backbone.
            self.angles=nn.Parameter(torch.zeros(self.stages,self.groups,group//2))
        elif kind=='dense':
            # Cayley parametrization is an exact differentiable orthogonal control.
            self.raw=nn.Parameter(torch.zeros(self.groups,group,group))
        elif kind=='smooth_ns5':
            self.raw=nn.Parameter(self.had.clone())
        self.cached_q=None

    def pair_rotation(self,x,stage,angles):
        # x [T,B,groups,group,H,W], with disjoint pairs at butterfly distance.
        stride=1<<stage
        z=x.reshape(*x.shape[:3],self.group//(2*stride),2,stride,*x.shape[-2:])
        a=z[:,:,:,:,0];b=z[:,:,:,:,1]
        theta=angles.reshape(self.groups,self.group//(2*stride),stride)
        c=theta.cos().to(x.dtype)[None,None,:,:,:,None,None]
        s=theta.sin().to(x.dtype)[None,None,:,:,:,None,None]
        return torch.stack((c*a-s*b,s*a+c*b),dim=4).reshape_as(x)

    def matrix(self):
        if not self.training and self.cached_q is not None: return self.cached_q
        eye=torch.eye(self.group,device=self.had.device,dtype=self.had.dtype).expand(self.groups,-1,-1)
        if self.kind=='identity': q=eye
        elif self.kind=='hadamard': q=self.had
        elif self.kind=='givens':
            q=eye.reshape(1,1,self.groups,self.group,self.group,1)
            for i in range(self.stages): q=self.pair_rotation(q,i,self.angles[i])
            q=q.reshape(self.groups,self.group,self.group)
        elif self.kind=='dense':
            a=self.raw-self.raw.transpose(-1,-2)
            q=torch.linalg.solve(eye-a,eye+a)
        elif self.kind=='smooth_ns5':
            q=self.raw/(self.raw.norm(dim=(-2,-1),keepdim=True)+1e-7)
            for _ in range(5):
                a=q@q.transpose(-1,-2)
                q=3.4445*q+(-4.775*a+2.0315*a@a)@q
        else: raise ValueError(self.kind)
        return q

    def forward(self,x):
        if self.kind=='identity': return x
        z=x.reshape(*x.shape[:2],self.groups,self.group,*x.shape[-2:])
        if self.kind=='givens':
            for i in range(self.stages): z=self.pair_rotation(z,i,self.angles[i])
        elif self.kind=='hadamard':
            for i in range(self.stages):
                stride=1<<i
                p=z.reshape(*z.shape[:3],self.group//(2*stride),2,stride,*z.shape[-2:])
                a=p[:,:,:,:,0];b=p[:,:,:,:,1]
                z=(torch.stack((a+b,a-b),dim=4)/math.sqrt(2)).reshape_as(z)
        else:
            z=torch.einsum('gij,tbgjhw->tbgihw',self.matrix().to(z.dtype),z)
        return z.reshape_as(x)

    def encoding_ops(self,elements):
        if self.kind=='identity': return (0,0)
        if self.kind=='givens': return (elements*2*self.stages,elements*self.stages)
        if self.kind=='hadamard': return (elements*self.stages,elements*self.stages)
        return (elements*self.group,elements*(self.group-1))

class CodedSpike(nn.Module):
    def __init__(self,neuron,channels,outputs,group,kind,learn_threshold=False):
        super().__init__()
        self.neuron=neuron
        self.rotation=ChannelRotation(channels,group,kind)
        self.outputs=outputs
        if learn_threshold: self.log_threshold=nn.Parameter(torch.zeros(channels))
        else: self.register_parameter('log_threshold',None)
        self.cost=None; self.capacity=0;self.firing=None;self.collect=False;self.diagnostics={}

    def forward(self,x):
        z=self.rotation(x)
        # Hard-reset LIF: positive input scaling is equivalent to threshold scaling.
        if self.log_threshold is not None:
            z=z/self.log_threshold.clamp(-2,2).exp()[None,None,:,None,None]
        s=self.neuron(z)
        self.cost=s.float().sum()*self.outputs/x.shape[1]
        self.capacity=s.numel()*self.outputs/x.shape[1]
        self.firing=s.float().mean()
        if self.collect:
            self.diagnostics={'spikes':s.detach().float().sum().item()/x.shape[1],
                'synops':self.cost.detach().item(),'capacity':self.capacity,
                'saturation':s.detach().bool().all(0).float().mean().item(),
                'silence':(~s.detach().bool().any(0)).float().mean().item(),
                'input_rms':x.detach().float().square().mean().sqrt().item(),
                'coded_rms':z.detach().float().square().mean().sqrt().item(),
                'input_max':x.detach().abs().max().item(),'coded_max':z.detach().abs().max().item(),
                'encoder_multiply':self.rotation.encoding_ops(x.numel()/x.shape[1])[0],
                'encoder_add':self.rotation.encoding_ops(x.numel()/x.shape[1])[1]}
        return s

class FoldedDecoder(nn.Module):
    def __init__(self,conv,rotation):
        super().__init__()
        self.conv=conv
        object.__setattr__(self,'rotation_ref',weakref.ref(rotation))
        self.cached_weight=None
    def weight(self):
        if not self.training and self.cached_weight is not None: return self.cached_weight
        q=self.rotation_ref().matrix()
        w=self.conv.weight[:,:,0,0]
        w=w.reshape(w.shape[0],q.shape[0],q.shape[1])
        return torch.einsum('ogj,gkj->ogk',w,q.to(w.dtype)).reshape_as(self.conv.weight)
    def forward(self,s):
        y=F.conv2d(s.flatten(0,1),self.weight(),self.conv.bias)
        return y.reshape(*s.shape[:2],*y.shape[1:])

def install(model,kind='identity',group=32,learn_threshold=False):
    from models.spikingresformer import GWFFN
    sites=[]
    for name,block in list(model.named_modules()):
        if isinstance(block,GWFFN):
            conv=block.down[1]
            sn=CodedSpike(block.down[0],conv.in_channels,conv.out_channels,group,kind,learn_threshold)
            block.down[0]=sn
            block.down[1]=FoldedDecoder(conv,sn.rotation)
            sites.append((name+'.down.0',sn))
    return sites

def clear_costs(sites):
    for _,site in sites: site.cost=None;site.firing=None

def prepare_inference(model):
    for m in model.modules():
        if isinstance(m,ChannelRotation): m.cached_q=None
        if isinstance(m,FoldedDecoder): m.cached_weight=None
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m,ChannelRotation): m.cached_q=m.matrix().detach()
        for m in model.modules():
            if isinstance(m,FoldedDecoder): m.cached_weight=m.weight().detach()

def clear_cache(model):
    for m in model.modules():
        if isinstance(m,ChannelRotation): m.cached_q=None
        if isinstance(m,FoldedDecoder): m.cached_weight=None
