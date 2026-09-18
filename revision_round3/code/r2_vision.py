"""Finite-budget vision adaptation; train and validation only in this module."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.mixup import Mixup

ARCHIVE = Path(os.environ.get('R2_ARCHIVE', '/root/autodl-tmp/snn0907'))
WORK = Path(os.environ.get('R2_WORK', '/root/snn_revision_round2'))
VISION = ARCHIVE / 'direction1_budget_rotation_20260907'
sys.path.insert(0, str(VISION / 'code'))
import train as v

MEAN = (.5071, .4867, .4408)
STD = (.2675, .2565, .2761)
INIT = VISION / 'results/reconstructed_identity_p0.0/best.pt'


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def configure():
    torch.set_num_threads(int(os.environ.get('R2_THREADS', '4')))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(T, path=INIT, training=False):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    cfg = cp.get('initial_config', cp.get('config'))
    if cfg is None or 'dataset' not in cfg:
        cfg = torch.load(INIT, map_location='cpu', weights_only=False)['config']
    m, sites = v.build_model(cfg)
    changes = v.load_preserving_buffers(m, cp['model'])
    m.T = T
    m.train(training)
    if training: v.clear_cache(m)
    else:
        m.requires_grad_(False); v.prepare_inference(m)
    return m, sites, cfg, changes


def evaluate_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def validation_loader(workers=4, batch=64):
    split = json.loads((VISION / 'data/CIFAR100_split.json').read_text())
    base = datasets.CIFAR100(str(VISION / 'data'), train=True, transform=evaluate_transform())
    return DataLoader(Subset(base, split['validation']), batch_size=batch, shuffle=False,
                      num_workers=workers, pin_memory=True)


class PlannedSamples(Dataset):
    """Per-exposure transforms are reproducible across workers, horizons and resume."""
    def __init__(self, seed, updates, start=0):
        self.base = datasets.CIFAR100(str(VISION / 'data'), train=True)
        split = json.loads((VISION / 'data/CIFAR100_split.json').read_text())
        pool = np.array(split['train'], dtype=np.int64)
        rng = np.random.default_rng(seed)
        size = updates * 128
        self.indices = np.concatenate([rng.permutation(pool) for _ in range((size+len(pool)-1)//len(pool))])[:size]
        self.aug_seeds = np.random.default_rng(seed+1_000_000).integers(0, 2**31-1, size=size, dtype=np.int64)
        self.start = start * 128
        self.transform = create_transform(input_size=32, is_training=True, scale=(1.,1.),
             ratio=(1.,1.), hflip=.5, auto_augment='rand-m7-n1-mstd0.5-inc1', mean=MEAN, std=STD)

    def __len__(self): return len(self.indices)-self.start

    def __getitem__(self, position):
        pos = position+self.start
        seed = int(self.aug_seeds[pos])
        random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
        x, y = self.base[int(self.indices[pos])]
        return self.transform(x), y


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


def state_cpu(m):
    return {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}


def atomic_save(obj, path):
    tmp = path.with_suffix('.partial'); torch.save(obj, tmp); tmp.replace(path)


@torch.no_grad()
def assess_validation(m, sites, loader):
    before = rng_state()
    m.eval(); v.prepare_inference(m)
    losses=[]; correct=[]
    for x,y in loader:
        x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
        v.functional.reset_net(m); v.clear_costs(sites)
        logits=m(x).mean(0).float()
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite validation logits')
        losses.extend(nn.functional.cross_entropy(logits,y,reduction='none').cpu().tolist())
        correct.extend((logits.argmax(-1)==y).cpu().tolist())
    v.functional.reset_net(m); v.clear_costs(sites); v.clear_cache(m); m.train()
    restore_rng(before)
    return {'ce':float(np.mean(losses)), 'accuracy':float(np.mean(correct)), 'samples':len(losses),
            'precision':'fp32', 'batch':64, 'split':'validation'}


def train(args):
    configure()
    out=WORK/'results'/args.stage/args.tag; out.mkdir(parents=True,exist_ok=True)
    dest=WORK/'models'/args.stage/args.tag; dest.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        if done['step']>=args.steps:
            print('ALREADY_COMPLETE',args.tag,flush=True);return
        assert args.stage=='probe', 'A scientific endpoint cannot be extended by this entry point.'
    cp_path=dest/'resume.pt'
    m,sites,initial_cfg,changes=load_model(args.T,cp_path if cp_path.exists() else INIT,True)
    params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    start=0; elapsed=0.; records=[]
    v.seed_all(args.seed)
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        assert (cp['config']['T'],cp['config']['seed'],cp['config']['lr'])==(args.T,args.seed,args.lr)
        opt.load_state_dict(cp['optimizer']);start=cp['step'];elapsed=cp['training_seconds'];records=cp['evaluations']
        restore_rng(cp['rng']);del cp
    ds=PlannedSamples(args.seed,args.steps,start)
    # An explicit generator prevents DataLoader worker startup from changing model RNG.
    dl=DataLoader(ds,batch_size=args.microbatch,shuffle=False,num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0,drop_last=False,
        generator=torch.Generator().manual_seed(args.seed+3_000_000))
    np.savez_compressed(out/'sample_plan.npz',image_id=ds.indices,augmentation_seed=ds.aug_seeds)
    plan_id=hashlib.sha256(ds.indices.tobytes()+ds.aug_seeds.tobytes()).hexdigest()
    cfg={'T':args.T,'seed':args.seed,'lr':args.lr,'updates':args.steps,'stage':args.stage,
      'precision':'fp32','tf32':False,'effective_batch':128,'microbatch':args.microbatch,
      'accumulation':128//args.microbatch,'optimizer':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,
      'clip_norm':1.,'learning_rate_schedule':'constant','workers':args.workers,
      'mean':MEAN,'std':STD,'auto_augment':'rand-m7-n1-mstd0.5-inc1','hflip':.5,'scale':[1.,1.],'ratio':[1.,1.],
      'mixup_alpha':.8,'cutmix_alpha':1.,'mixup_prob':1.,'switch_prob':.5,'mixup_mode':'batch','label_smoothing':.1,
      'sample_plan_sha256':plan_id,'initial_checkpoint':str(INIT),'initial_config':initial_cfg,
      'initial_buffer_dtype_changes':changes,'mixup_seed_rule':'(seed*100003+step*1009+micro) modulo (2**32-1)','gpu':torch.cuda.get_device_name(),
      'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,
      'cuda':torch.version.cuda,'cpu_threads':torch.get_num_threads(),'validation_batch':64}
    write_json(out/'config.json',cfg)
    mixup=Mixup(mixup_alpha=.8,cutmix_alpha=1.,prob=1.,switch_prob=.5,mode='batch',label_smoothing=.1,num_classes=100)
    val=validation_loader(args.workers)
    steps_path=out/'steps.jsonl'
    if steps_path.exists():
        old=[line for line in steps_path.read_text().splitlines() if json.loads(line)['step']<=start]
        steps_path.write_text('\n'.join(old)+('\n' if old else ''))
    m.train();v.clear_cache(m);torch.cuda.reset_peak_memory_stats();iterator=iter(dl)
    assert 128%args.microbatch==0
    with steps_path.open('a',buffering=1) as log:
        for step in range(start+1,args.steps+1):
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True);loss_value=0.
            for micro in range(128//args.microbatch):
                x,y=next(iterator);assert len(y)==args.microbatch
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                mix_seed=(args.seed*100003+step*1009+micro)% (2**32-1)
                old_np=np.random.get_state();np.random.seed(mix_seed);x,target=mixup(x,y);np.random.set_state(old_np)
                logits=m(x).mean(0).float()
                loss=-(target*nn.functional.log_softmax(logits,-1)).sum(-1).mean()
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at step {step}')
                (loss/(128//args.microbatch)).backward();loss_value+=float(loss.detach())/(128//args.microbatch)
                v.functional.reset_net(m);v.clear_costs(sites)
            norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize();seconds=time.perf_counter()-tick;elapsed+=seconds
            row={'step':step,'loss':loss_value,'gradient_norm':float(norm),'seconds':seconds,
                 'training_seconds':elapsed,'examples':128,'sample_plan_start':(step-1)*128}
            log.write(json.dumps(row)+'\n')
            if step%64==0 or step==args.steps:
                write_json(out/'progress.json',{'status':'running',**row,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
                print(json.dumps({'tag':args.tag,**row}),flush=True)
            if step in [512,1024,2048] or step==args.steps:
                result=assess_validation(m,sites,val)
                records.append({'step':step,'validation':result,'training_seconds':elapsed})
                write_json(out/'evaluations.json',records)
                atomic_save({'model':m.state_dict(),'initial_config':initial_cfg,'config':cfg,'step':step},dest/f'step{step}.pt')
                print(json.dumps({'tag':args.tag,'step':step,'validation':result}),flush=True)
            if step%256==0 or step==args.steps:
                atomic_save({'model':m.state_dict(),'optimizer':opt.state_dict(),'initial_config':initial_cfg,
                    'config':cfg,'step':step,'rng':rng_state(),'training_seconds':elapsed,'evaluations':records},cp_path)
    write_json(out/'complete.json',{'status':'complete','step':args.steps,'training_seconds':elapsed,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'evaluations':records,
        'checkpoint':str(dest/f'step{args.steps}.pt'),'config':cfg})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--T',type=int,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--lr',type=float,required=True);ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--stage',choices=['probe','screen','formal'],required=True)
    ap.add_argument('--microbatch',type=int,default=64);ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args()
    try:train(args)
    except Exception:
        write_json(WORK/'results'/args.stage/args.tag/'failure.json',{'status':'failed','error':traceback.format_exc()})
        raise
