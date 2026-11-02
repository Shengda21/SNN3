"""Controlled full-backbone experiments. Test data are evaluated once, after validation selection."""
import os,sys,time,json,math,random,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/SpikingResformer'))
import numpy as np
# SpikingJelly 0.0.0.0.14 CUDA type checks use the removed NumPy alias.
# Restore its original builtin-int meaning; no neuron/math changes.
if 'int' not in np.__dict__: np.int=int
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
import torchvision
from torchvision import transforms
from timm.data import create_transform
from timm.data.mixup import Mixup
from spikingjelly.activation_based import functional
from models.spikingresformer import spikingresformer_cifar
from rotations import install,clear_costs,prepare_inference,clear_cache,ChannelRotation,FoldedDecoder,CodedSpike
from checkpoint_io import load_preserving_buffers

def save_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def build_model(cfg):
    # Re-seed model construction independent of data-loading and intervention parameters.
    seed_all(cfg['seed'])
    model=spikingresformer_cifar(num_classes=100 if cfg['dataset']=='CIFAR100' else 10,
                                img_size=32 if cfg['dataset']!='CIFAR10DVS' else 64,T=cfg['T'])
    sites=install(model,cfg['kind'],cfg.get('group',32),cfg.get('learn_threshold',False))
    return model.cuda(),sites

class DVSDataset(torch.utils.data.Dataset):
    def __init__(self,base,indices,train=False): self.base=base;self.indices=indices;self.train=train
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        x,y=self.base[self.indices[i]]
        x=torch.from_numpy(x).float()
        x=torch.nn.functional.interpolate(x,size=(64,64),mode='bilinear',align_corners=False)
        if self.train:
            if torch.rand(())<0.5: x=x.flip(-1)
            shift=torch.randint(-5,6,(2,))
            x=torch.roll(x,tuple(shift.tolist()),(-2,-1))
        x=torch.cat((x,x.sum(1,keepdim=True)),1)
        return x,y

def datasets(cfg):
    name=cfg['dataset']
    if name=='CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        base=CIFAR10DVS(str(ROOT/'data/CIFAR10DVS'),data_type='frame',frames_number=cfg['T'],split_by='number')
        split=json.loads((ROOT/'data/CIFAR10DVS_split.json').read_text())
        return tuple(DVSDataset(base,split[s],s=='train') for s in ['train','validation','test'])
    cls=getattr(torchvision.datasets,name)
    mean=(0.5071,0.4867,0.4408) if name=='CIFAR100' else (0.4914,0.4822,0.4465)
    std=(0.2675,0.2565,0.2761) if name=='CIFAR100' else (0.2470,0.2435,0.2616)
    train_tf=create_transform(input_size=32,is_training=True,scale=(1.,1.),ratio=(1.,1.),
        hflip=.5,auto_augment='rand-m7-n1-mstd0.5-inc1',mean=mean,std=std)
    eval_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    split=json.loads((ROOT/'data'/f'{name}_split.json').read_text())
    tr=Subset(cls(str(ROOT/'data'),train=True,transform=train_tf),split['train'])
    va=Subset(cls(str(ROOT/'data'),train=True,transform=eval_tf),split['validation'])
    te=cls(str(ROOT/'data'),train=False,transform=eval_tf)
    return tr,va,te

def loader(ds,cfg,train=False):
    generator=torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=train,num_workers=cfg.get('workers',6),
        pin_memory=True,drop_last=train,persistent_workers=cfg.get('workers',6)>0,generator=generator)

@torch.no_grad()
def evaluate(model,sites,dl,max_batches=None):
    model.eval();prepare_inference(model)
    correct=total=0;loss_sum=cost_sum=0.;start=time.time()
    for step,(x,y) in enumerate(dl):
        if max_batches is not None and step>=max_batches:break
        x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16): logits=model(x).mean(0).float()
        if not torch.isfinite(logits).all(): raise RuntimeError('Non-finite evaluation logits')
        loss_sum+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(-1)==y).sum().item();total+=len(y)
        cost_sum+=sum(s.cost.item() for _,s in sites)*len(y)
        functional.reset_net(model);clear_costs(sites)
    return {'accuracy':100*correct/total,'loss':loss_sum/total,'selected_synops_per_sample':cost_sum/total,
            'samples':total,'seconds':time.time()-start}

@torch.no_grad()
def profile(model,sites,dl,max_batches=8):
    """Count spike-weight accumulations, analog MACs, and online encoder ops separately."""
    from models.submodules.layers import SpikingMatmul
    model.eval();prepare_inference(model)
    totals={'synops':0.,'analog_mac':0.,'encoder_multiply':0.,'encoder_add':0.}
    hooks=[];samples=0;layers={};batch_size=1
    def conv_hook(name,m,x,y):
        nonlocal batch_size
        v=x[0].detach();w=m.conv if isinstance(m,FoldedDecoder) else m
        v=v.flatten(0,1) if v.ndim==5 else v
        if name=='prologue.0':
            mac=y.numel()*(w.in_channels//w.groups)*math.prod(w.kernel_size)
            totals['analog_mac']+=mac;return
        if not bool(((v==0)|(v==1)).all()): raise RuntimeError('Nonbinary convolution input '+name)
        # Count padding/stride exactly by applying the spatial kernel to summed input channels.
        # Fan-out is equal for each input within a convolution group.
        summed=v.float().sum(1,keepdim=True)
        ones=torch.ones(1,1,*w.kernel_size,device=v.device)
        count=nn.functional.conv2d(summed,ones,stride=w.stride,padding=w.padding,dilation=w.dilation).sum()
        count=count.item()*(w.out_channels//w.groups)
        totals['synops']+=count;layers[name]=layers.get(name,0.)+count
    def mat_hook(name,m,x,y):
        left,right=x
        binary=right if m.spike=='r' else left
        if not bool(((binary==0)|(binary==1)).all()):raise RuntimeError('Nonbinary matmul input '+name)
        n=right.detach().float().sum().item()*left.shape[-2] if m.spike=='r' else left.detach().float().sum().item()*right.shape[-1]
        totals['synops']+=n;layers[name]=layers.get(name,0.)+n
    skip=set(n+'.conv' for n,m in model.named_modules() if isinstance(m,FoldedDecoder))
    for name,m in model.named_modules():
        if name in skip: continue
        if isinstance(m,(nn.Conv2d,FoldedDecoder)):
            hooks.append(m.register_forward_hook(lambda m,x,y,n=name:conv_hook(n,m,x,y)))
        elif isinstance(m,SpikingMatmul):
            hooks.append(m.register_forward_hook(lambda m,x,y,n=name:mat_hook(n,m,x,y)))
        elif isinstance(m,nn.Linear):
            hooks.append(m.register_forward_hook(lambda m,x,y:totals.__setitem__('analog_mac',totals['analog_mac']+y.numel()*m.in_features)))
    diagnostic_sums={name:{} for name,_ in sites}
    for _,site in sites:site.collect=True
    try:
        for i,(x,y) in enumerate(dl):
            if i>=max_batches:break
            x=x.cuda();batch_size=len(y)
            with torch.autocast('cuda',dtype=torch.float16):model(x)
            samples+=len(y)
            for name,site in sites:
                d=site.diagnostics
                for key,val in d.items():diagnostic_sums[name][key]=diagnostic_sums[name].get(key,0.)+val*len(y)
                totals['encoder_multiply']+=d['encoder_multiply']*len(y)
                totals['encoder_add']+=d['encoder_add']*len(y)
            functional.reset_net(model);clear_costs(sites)
    finally:
        for h in hooks:h.remove()
        for _,site in sites:site.collect=False
    totals={k:v/samples for k,v in totals.items()}
    # Transparent equal-operation proxy. This is not measured chip energy.
    totals['total_arithmetic_proxy']=sum(totals.values())
    totals['profile_samples']=samples
    totals['layer_synops']={k:v/samples for k,v in layers.items()}
    totals['sites']={k:{a:b/samples for a,b in v.items()} for k,v in diagnostic_sums.items()}
    totals['orthogonality_error']={name:float((s.rotation.matrix().transpose(-1,-2)@s.rotation.matrix()-torch.eye(s.rotation.group,device='cuda')).abs().max()) for name,s in sites}
    return totals

def main(cfg):
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    out=ROOT/'results'/cfg['id'];out.mkdir(parents=True,exist_ok=True)
    save_json(out/'config.json',cfg)
    model,sites=build_model(cfg)
    ds_train,ds_val,ds_test=datasets(cfg)
    dl_train=loader(ds_train,cfg,True);dl_val=loader(ds_val,cfg)
    # Same shared parameters and random augmentation seeds in each paired run.
    seed_all(cfg['seed'])
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and not any(k in n for k in ['angles','rotation.raw','log_threshold'])],'weight_decay':cfg.get('weight_decay',.01)},
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and any(k in n for k in ['angles','rotation.raw','log_threshold'])],'weight_decay':0.}],lr=cfg.get('lr',5e-4))
    scaler=torch.amp.GradScaler('cuda')
    best=-1.;start_epoch=0;total_seconds=0.;peak_vram=0
    if (out/'last.pt').exists():
        cp=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        load_preserving_buffers(model,cp['model']);optimizer.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        best=cp['best'];start_epoch=cp['epoch']+1;total_seconds=cp.get('total_seconds',0)
        if 'rng' in cp:
            random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy']);torch.set_rng_state(cp['rng']['torch'].cpu());torch.cuda.set_rng_state_all(cp['rng']['cuda'])
        del cp
    mixup=Mixup(mixup_alpha=.8,cutmix_alpha=1.,prob=1.,switch_prob=.5,mode='batch',label_smoothing=.1,
                num_classes=100 if cfg['dataset']=='CIFAR100' else 10) if cfg['dataset']!='CIFAR10DVS' else None
    save_json(out/'status.json',{'state':'running','epoch':start_epoch,'total_epochs':cfg['epochs'],'pid':os.getpid()})
    for epoch in range(start_epoch,cfg['epochs']):
        epoch_start=time.time();model.train();clear_cache(model)
        # Epoch-specific seed makes resumed augmentations and shuffles auditable; worker RNG streams are not bitwise restored.
        lr=cfg.get('lr',5e-4)*(.01+.99*.5*(1+math.cos(math.pi*epoch/max(1,cfg['epochs']-1))))
        if epoch<5:lr=cfg.get('lr',5e-4)*(epoch+1)/5
        for g in optimizer.param_groups:g['lr']=lr
        n=0;sum_loss=sum_penalty=0.;correct=0
        optimizer.zero_grad(set_to_none=True)
        train_steps=min(len(dl_train),cfg.get('max_train_batches') or len(dl_train))
        accumulation=cfg.get('accumulation',1)
        for step,(x,y) in enumerate(dl_train):
            if cfg.get('max_train_batches') and step>=cfg['max_train_batches']:break
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            target=y
            if mixup is not None:x,target=mixup(x,y)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x).mean(0).float()
                task=-(target*nn.functional.log_softmax(logits,-1)).sum(-1).mean() if target.ndim==2 else nn.functional.cross_entropy(logits,target,label_smoothing=.1)
                if cfg.get('unweighted',False):penalty=torch.stack([s.firing for _,s in sites]).mean()
                else:penalty=sum(s.cost for _,s in sites)/sum(s.capacity for _,s in sites)
                strength=cfg.get('penalty',0.)*min(1.,(epoch+1)/10.)
                loss=task+strength*penalty
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            group_start=(step//accumulation)*accumulation
            group_count=min(accumulation,train_steps-group_start)
            scaler.scale(loss/group_count).backward()
            if (step+1)%accumulation==0 or step+1==train_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update()
                optimizer.zero_grad(set_to_none=True)
            n+=len(y);sum_loss+=task.item()*len(y);sum_penalty+=penalty.item()*len(y)
            correct+=(logits.argmax(-1)==y).sum().item()
            functional.reset_net(model);clear_costs(sites)
        val=evaluate(model,sites,dl_val,cfg.get('max_eval_batches'))
        seconds=time.time()-epoch_start;total_seconds+=seconds
        peak_vram=max(peak_vram,torch.cuda.max_memory_allocated())
        record={'epoch':epoch+1,'train_loss':sum_loss/n,'train_spike_cost_fraction':sum_penalty/n,'train_accuracy_unmixed_labels':100*correct/n,
                'validation':val,'lr':lr,'epoch_seconds':seconds,'cumulative_seconds':total_seconds,'peak_vram_bytes':peak_vram}
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps({'id':cfg['id'],**record}),flush=True)
        improved=val['accuracy']>best
        if improved:best=val['accuracy']
        cp={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'config':cfg,'total_seconds':total_seconds,
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
        torch.save(cp,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        if improved:torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch},out/'best.pt')
        save_json(out/'status.json',{'state':'running','epoch':epoch+1,'total_epochs':cfg['epochs'],'best_validation_accuracy':best,
                                  'last_epoch_seconds':seconds,'pid':os.getpid()})
    cp=torch.load(out/'best.pt',map_location='cuda',weights_only=False);load_preserving_buffers(model,cp['model']);del cp
    model.eval();prepare_inference(model)
    # Screening runs never read the test split. Full experiments select on validation only.
    final_loader=dl_val if cfg['phase']!='full' else loader(ds_test,cfg)
    final=evaluate(model,sites,final_loader,cfg.get('max_eval_batches'))
    perf=profile(model,sites,final_loader,max_batches=cfg.get('profile_batches',8))
    latency=[]
    x,_=next(iter(final_loader));x=x[:min(len(x),8)].cuda()
    for i in range(25):
        functional.reset_net(model);clear_costs(sites)
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):model(x)
        torch.cuda.synchronize()
        if i>=5:latency.append((time.perf_counter()-t)*1000)
    summary={'id':cfg['id'],'phase':cfg['phase'],'config':cfg,'final_split':'test' if cfg['phase']=='full' else 'validation',
             'accuracy':final['accuracy'],'best_validation_accuracy':best,'profile':perf,'train_seconds':total_seconds,
             'parameters':sum(p.numel() for p in model.parameters()),'latency_ms_median':float(np.median(latency)),
             'latency_batch_size':len(x),'peak_vram_bytes':peak_vram,'energy_measured':False}
    save_json(out/'summary.json',summary)
    save_json(out/'status.json',{'state':'complete','epoch':cfg['epochs'],'accuracy':final['accuracy'],'phase':cfg['phase']})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    try:main(cfg)
    except Exception:
        out=ROOT/'results'/cfg['id'];out.mkdir(parents=True,exist_ok=True)
        save_json(out/'status.json',{'state':'failed','error':traceback.format_exc()})
        raise
