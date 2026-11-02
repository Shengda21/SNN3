"""Shared real model/data operations. GPU imports happen only when called."""
import gc
import json
from pathlib import Path
import random
import sys
import numpy as np
from common import BUNDLE, environment, protocol, read

MEAN = (.5071,.4867,.4408)
STD = (.2675,.2565,.2761)


def configure():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vision_root(root):
    return Path(root)/'archive/direction1_budget_rotation_20260907'


def build_model(root, architecture, T, seed):
    import torch
    environment(root)
    if architecture == 'resformer':
        sys.path.insert(0,str(Path(root)/'legacy/code'))
        import r2_vision as rv
        cfg = read(vision_root(root)/'configs/screen_identity_p0.0.json')
        cfg.update(seed=seed,T=T)
        model,sites = rv.v.build_model(cfg)
        def reset_fn():
            rv.v.functional.reset_net(model)
            rv.v.clear_costs(sites)
        metadata = {'architecture':'resformer','source':'bundled official SpikingResformer with historical identity-only execution wrappers',
                    'constructor_seed':seed,'added_trainable_parameters':False}
        object.__setattr__(model,'_revision_rv',rv)
    elif architecture == 'spikformer':
        from backbone import build_model as build_f
        model,reset_fn,metadata = build_f('spikformer',T,seed,device='cuda')
    else:
        raise ValueError(architecture)
    object.__setattr__(model,'_revision_reset_fn',reset_fn)
    object.__setattr__(model,'_revision_architecture',architecture)
    object.__setattr__(model,'_revision_metadata',metadata)
    return model


def set_window(model,T):
    if model._revision_architecture == 'resformer':
        model.T = int(T)
    else:
        from backbone import set_window as set_f
        set_f(model,T)


def reset(model):
    model._revision_reset_fn()


def prepare_mode(model,training):
    model.train(training)
    if model._revision_architecture == 'resformer':
        if training:
            model._revision_rv.v.clear_cache(model)
        else:
            model._revision_rv.v.prepare_inference(model)
    reset(model)


def forward(model,x):
    reset(model)
    result = model(x)
    if result.ndim == 3:
        result = result.mean(0)
    assert result.ndim == 2 and result.shape[-1] == 100, tuple(result.shape)
    return result.float()


def load_model(root,architecture,checkpoint,T):
    import torch
    cp = torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg = cp.get('initial_config',cp.get('config',{}))
    if 'initial_config' in cfg and 'dataset' not in cfg:
        cfg = cfg['initial_config']
    seed = cfg.get('construction_seed',cfg.get('seed',12450))
    model = build_model(root,architecture,T,seed)
    if architecture == 'resformer':
        changes = model._revision_rv.v.load_preserving_buffers(model,cp['model'])
    else:
        model.load_state_dict(cp['model'],strict=True)
        changes = []
    object.__setattr__(model,'_revision_loaded_dtype_changes',changes)
    prepare_mode(model,False)
    return model


def evaluate_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def training_transform():
    from timm.data import create_transform
    return create_transform(input_size=32,is_training=True,scale=(1.,1.),ratio=(1.,1.),
        hflip=.5,auto_augment='rand-m7-n1-mstd0.5-inc1',mean=MEAN,std=STD)


def split(root):
    return read(vision_root(root)/'data/CIFAR100_split.json')


def sample_plan(root,seed,updates):
    pool = np.asarray(split(root)['train'],dtype=np.int64)
    count = updates*128
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.permutation(pool) for _ in range((count+len(pool)-1)//len(pool))])[:count]
    aug = np.random.default_rng(seed+1_000_000).integers(0,2**31-1,size=count,dtype=np.int64)
    return indices,aug


def planned_dataset(root,seed,updates,start=0):
    import torch
    from torch.utils.data import Dataset
    from torchvision.datasets import CIFAR100
    class Planned(Dataset):
        def __init__(self):
            self.base = CIFAR100(str(vision_root(root)/'data'),train=True)
            self.indices,self.aug_seeds = sample_plan(root,seed,updates)
            self.transform = training_transform()
            self.start = start*128
        def __len__(self):
            return len(self.indices)-self.start
        def __getitem__(self,index):
            index += self.start
            item_seed = int(self.aug_seeds[index])
            random.seed(item_seed)
            np.random.seed(item_seed)
            torch.random.default_generator.manual_seed(item_seed)
            image,label = self.base[int(self.indices[index])]
            return self.transform(image),label
    return Planned()


def construction_loaders(root,cohort,epoch=0):
    import torch
    from torch.utils.data import DataLoader,Subset
    from torchvision.datasets import CIFAR100
    c = protocol()['cohorts'][cohort]
    ds = CIFAR100(str(vision_root(root)/'data'),train=True,transform=training_transform())
    epoch_seed = c['construction_seed']+epoch*100003
    train = DataLoader(Subset(ds,split(root)['train']),batch_size=c['construction_batch'],
        shuffle=True,drop_last=True,num_workers=6,pin_memory=True,
        generator=torch.Generator().manual_seed(epoch_seed),persistent_workers=False)
    return train,validation_loader(root)


def validation_loader(root,batch=64):
    from torch.utils.data import DataLoader,Subset
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(str(vision_root(root)/'data'),train=True,transform=evaluate_transform())
    return DataLoader(Subset(ds,split(root)['validation']),batch_size=batch,shuffle=False,
                      num_workers=4,pin_memory=True)


def calibration_images(root,seed=9640):
    import torch
    from torchvision.datasets import CIFAR100
    ids = np.random.default_rng(seed).choice(np.asarray(split(root)['train']),2048,replace=False)
    ds = CIFAR100(str(vision_root(root)/'data'),train=True,transform=evaluate_transform())
    return torch.stack([ds[int(i)][0] for i in ids]),ids


def calibrate(root,model,seed=9640):
    from normalization import calibrate_bn
    images,ids = calibration_images(root,seed)
    prepare_mode(model,False)
    bank,report = calibrate_bn(model,images,lambda:reset(model),lambda x:forward(model,x),batch=64)
    report['train_ids'] = ids.tolist()
    report['calibration_seed'] = seed
    # Common audit keys used by selection and replay gates.
    report['samples'] = len(ids)
    report['batches'] = len(ids)//64
    report['protected_values_and_dtypes_unchanged'] = True  # calibrate_bn raises if protection fails
    prepare_mode(model,False)
    return bank,report


def evaluate_validation(root,model):
    import torch
    from torch.nn import functional as F
    prepare_mode(model,False)
    loss,correct,count = 0.,0,0
    with torch.no_grad():
        for x,y in validation_loader(root):
            x,y = x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            z = forward(model,x)
            assert torch.isfinite(z).all()
            loss += float(F.cross_entropy(z,y,reduction='sum'))
            correct += int((z.argmax(-1)==y).sum())
            count += len(y)
    reset(model)
    assert count == 5000
    return {'ce':loss/count,'accuracy':correct/count,'samples':count,'split':'validation'}


def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(data,path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    torch.save(data,temporary)
    temporary.replace(path)
