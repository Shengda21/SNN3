"""Independent initialization, retaining the original 30-epoch training loop.
The original post-training test/profile/timing operations are omitted.
"""
import os,json
from pathlib import Path
import r2_vision as rv
from train import *
ROOT=rv.WORK/'initialization'
ROOT.mkdir(parents=True,exist_ok=True)
if not (ROOT/'data').exists():(ROOT/'data').symlink_to(rv.VISION/'data',target_is_directory=True)
rv.v.ROOT=ROOT
def build_initial(cfg):
    torch.set_num_threads(4)
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
    save_json(out/'complete.json',{'status':'complete','epochs':cfg['epochs'],'seed':cfg['seed'],
        'best_validation_accuracy':best,'training_seconds':total_seconds,'checkpoint':str(out/'best.pt'),
        'source':'independent random initialization; original AMP/TF32 initialization procedure',
        'post_training_test_profile_timing_omitted':True})
    save_json(out/'status.json',{'state':'complete','epoch':cfg['epochs'],'total_epochs':cfg['epochs'],
        'best_validation_accuracy':best,'training_seconds':total_seconds,'pid':os.getpid()})

if __name__=='__main__':
    cfg=json.loads((rv.VISION/'results/reconstructed_identity_p0.0/config.json').read_text())
    cfg.update(id='independent12550',seed=12550,workers=6)
    build_initial(cfg)
