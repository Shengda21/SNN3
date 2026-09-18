import gc, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
# SpikingJelly's pinned CuPy bridge uses the removed alias; its documented
# historical meaning is exactly Python int. Keep the compatibility local.
np.int = int
import torch
from torch import nn
from torch.nn.utils import parametrize
from safetensors.torch import load_file
from transformers import BertConfig, BertTokenizerFast

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/SmoothSpike'
sys.path.insert(0,str(VENDOR))
import spikingbert_rot as original
import spikingbert_rot_inf as fused
import convertor
NS5=original.msign
original._RUNTIME_DEVICE=torch.device('cpu')
fused._RUNTIME_DEVICE=torch.device('cpu')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def config():
    c=BertConfig.from_pretrained(str(VENDOR/'tokenizer_files'),local_files_only=True)
    c.T=4;c._attn_implementation='eager';c.use_cache=False
    return c

def dct(n,seed,device='cpu'):
    k=torch.arange(n,dtype=torch.float64)[:,None]
    j=torch.arange(n,dtype=torch.float64)[None,:]
    C=(2/n)**0.5*torch.cos(math.pi*k*(2*j+1)/(2*n));C[0]/=math.sqrt(2)
    g=torch.Generator().manual_seed(seed)
    signs=torch.randint(0,2,(n,),generator=g,dtype=torch.int64)*2-1
    return (C*signs).to(device=device,dtype=torch.float32)

class BlockCayley(nn.Module):
    def __init__(self,base,block=16):
        super().__init__();self.squeeze=base.ndim==2;self.block=block
        self.register_buffer('base',base.unsqueeze(0) if self.squeeze else base)
    def right_inverse(self,q):
        return q.new_zeros((self.base.shape[0],self.base.shape[-1]//self.block,self.block,self.block))
    def forward(self,t):
        a=(t-t.mT)/2
        I=torch.eye(self.block,device=t.device,dtype=t.dtype)
        r=torch.linalg.solve(I-a,I+a)
        base=self.base.reshape(self.base.shape[0],-1,self.block,self.base.shape[-1])
        q=(r@base).reshape_as(self.base)
        return q[0] if self.squeeze else q

@torch.no_grad()
def rebase_local(model):
    # Effective OLD matrices are evaluated exactly as the FP32 training code.
    for li,layer in enumerate(model.bert.encoder.layer):
        qold=NS5(layer.attention.H2).double()
        qnew=dct(qold.shape[-1],7000+li).to(qold.device).double()
        v=layer.attention.self.value;o=layer.attention.output.dense
        v.weight.copy_((qnew@qold.T@v.weight.double()).float())
        if v.bias is not None:v.bias.copy_((qnew@qold.T@v.bias.double()).float())
        o.weight.copy_((o.weight.double()@qold@qnew.T).float())
        layer.attention.H2.copy_(qnew.float())
        parametrize.register_parametrization(layer.attention,'H2',BlockCayley(qnew.float()),unsafe=True)
        old=NS5(layer.H3).double(); k,n,_=old.shape
        new=torch.stack([dct(n,7100+li*4+i) for i in range(k)]).to(old.device).double()
        v=layer.intermediate.dense;o=layer.output.dense
        vin=v.weight.double().reshape(k,n,-1)
        v.weight.copy_((new@old.mT@vin).reshape_as(v.weight).float())
        if v.bias is not None:v.bias.copy_((new@old.mT@v.bias.double().reshape(k,n,1)).reshape_as(v.bias).float())
        wo=o.weight.double().reshape(o.weight.shape[0],k,n).permute(1,0,2)
        o.weight.copy_((wo@old@new.mT).permute(1,0,2).reshape_as(o.weight).float())
        layer.H3.copy_(new.float())
        parametrize.register_parametrization(layer,'H3',BlockCayley(new.float()),unsafe=True)

def dispatch(model,variant,cache_all=False):
    h1=model.bert.H1
    with torch.no_grad():q1=NS5(h1).detach()
    cache={}
    def mapping(g,*args,**kwargs):
        if g is h1:return q1
        if variant.startswith('dct_'):return g
        if cache_all:
            key=g.data_ptr()
            if key not in cache:cache[key]=NS5(g).detach()
            return cache[key]
        return NS5(g,*args,**kwargs)
    module=fused if variant=='folded' else original
    module.msign=mapping
    return cache

def make_model(variant,device='cuda'):
    original.msign=NS5;fused.msign=NS5
    seed_all(4000)
    state=load_file(str(ROOT/'data/model.safetensors'),device='cpu')
    if variant=='folded':
        cpath=ROOT/'data/fused_state.pt'
        if cpath.exists():state=torch.load(cpath,map_location='cpu',weights_only=True)
        else:
            state=convertor.fuse_rotation_matrices_in_state_dict(state,config())
            torch.save(state,cpath)
        model=fused.BertForMaskedLM(config())
    else:model=original.BertForMaskedLM(config())
    keys=model.load_state_dict(state,strict=False)
    allowed={'cls.predictions.decoder.weight','cls.predictions.decoder.bias'}
    if set(keys.missing_keys)-allowed or keys.unexpected_keys:
        raise RuntimeError(str(keys))
    model.tie_weights();del state
    model=model.to(device)
    if variant.startswith('dct_'):
        rebase_local(model)
        if variant=='dct_fixed':
            for layer in model.bert.encoder.layer:
                parametrize.remove_parametrizations(layer.attention,'H2',leave_parametrized=True)
                parametrize.remove_parametrizations(layer,'H3',leave_parametrized=True)
                layer.attention.H2.requires_grad_(False);layer.H3.requires_grad_(False)
    model.bert.H1.requires_grad_(False)
    dispatch(model,variant)
    gc.collect()
    return model

def prepare_data():
    path=ROOT/'data/tokenized.pt'
    if path.exists():return torch.load(path,weights_only=True)
    import pyarrow.parquet as pq
    tok=BertTokenizerFast.from_pretrained(str(VENDOR/'tokenizer_files'),local_files_only=True)
    arrays={}
    for split in ['train','validation']:
        texts=pq.read_table(ROOT/f'data/{split}.parquet',columns=['text']).to_pydict()['text']
        ids=[]
        for start in range(0,len(texts),256):
            encoded=tok(texts[start:start+256],add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)
            for row in encoded['input_ids']:ids.extend(row)
        length=62;count=len(ids)//length
        middle=torch.tensor(ids[:count*length],dtype=torch.long).reshape(count,length)
        arrays[split]=torch.cat([torch.full((count,1),tok.cls_token_id),middle,torch.full((count,1),tok.sep_token_id)],1)
    g=torch.Generator().manual_seed(6101)
    order=torch.randperm(len(arrays['validation']),generator=g)
    arrays['tune']=arrays['validation'][order[:64]]
    arrays['heldout']=arrays['validation'][order[64:320]]
    arrays['tune_indices']=order[:64];arrays['heldout_indices']=order[64:320]
    arrays['mask_token_id']=torch.tensor(tok.mask_token_id)
    arrays['vocab_size']=torch.tensor(tok.vocab_size)
    torch.save(arrays,path)
    save_json(ROOT/'environment/data.json',{'source':'Salesforce/wikitext wikitext-2-raw-v1','train_chunks':len(arrays['train']),
        'validation_chunks':len(arrays['validation']),'tune_n':64,'heldout_n':256,'length':64,'split_seed':6101,
        'tune_indices':order[:64].tolist(),'heldout_indices':order[64:320].tolist(),'test_used':False})
    return arrays

def mask_batch(x,g,mask_id=103,vocab=30522):
    x=x.clone();labels=x.clone()
    selected=torch.rand(x.shape,generator=g)<.15
    selected[:,0]=False;selected[:,-1]=False
    labels[~selected]=-100
    types=torch.rand(x.shape,generator=g)
    x[selected&(types<.8)]=mask_id
    random_ids=torch.randint(vocab,x.shape,generator=g)
    take=selected&(types>=.8)&(types<.9);x[take]=random_ids[take]
    return x,labels

def fixed_split(data,split):
    return mask_batch(data[split],torch.Generator().manual_seed(6100+(split=='heldout')))

@torch.no_grad()
def evaluate(model,data,split='heldout',batch=4,save_prefix=None):
    model.eval();xs,ys=fixed_split(data,split)
    losses=[];counts=[];correct=[];preds=[];refs=[]
    for start in range(0,len(xs),batch):
        x=xs[start:start+batch].cuda();y=ys[start:start+batch].cuda()
        logits=model(input_ids=x,attention_mask=torch.ones_like(x)).logits.float()
        loss=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').reshape_as(y)
        p=logits.argmax(-1);mask=y!=-100
        losses.extend(loss.sum(-1).cpu().tolist());counts.extend(mask.sum(-1).cpu().tolist())
        correct.extend(((p==y)&mask).sum(-1).cpu().tolist())
        preds.extend(p[mask].cpu().tolist());refs.extend(y[mask].cpu().tolist())
    stats={'loss':sum(losses)/sum(counts),'masked_accuracy':sum(correct)/sum(counts),'masked_tokens':sum(counts),'sequences':len(xs)}
    if save_prefix:
        np.savez_compressed(str(save_prefix)+'.npz',loss_sum=losses,counts=counts,correct=correct,predictions=preds,labels=refs)
    return stats

def parameters(model):
    ordinary=[];rot=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.endswith('.H2') or name.endswith('.H3') or '.parametrizations.H' in name:rot.append(p)
        else:ordinary.append(p)
    return ordinary,rot

class SpikeStats:
    def __init__(self,model):
        from spikingjelly.activation_based.neuron import LIFNode
        self.rows={};self.handles=[]
        for name,m in model.named_modules():
            if isinstance(m,LIFNode):
                self.handles.append(m.register_forward_hook(self.hook(name)))
    def hook(self,name):
        def call(module,inputs,out):
            if out.ndim!=4:return
            s=out.detach().float().sum(0)
            vals=torch.stack([(s==out.shape[0]).sum(),(s==0).sum(),s.sum(),torch.as_tensor(s.numel(),device=s.device)])
            self.rows[name]=self.rows.get(name,0)+vals
        return call
    def finish(self):
        result={}
        for name,v in self.rows.items():
            sat,silent,spikes,n=v.cpu().tolist()
            result[name]={'saturated_fraction':sat/n,'silent_fraction':silent/n,'firing_rate':spikes/(4*n),'neurons':n}
        for h in self.handles:h.remove()
        return result
