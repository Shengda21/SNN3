import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[2]))
"""One explicitly requested construction/adaptation trajectory; never launched by plan/prepare."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from common import BUNDLE, environment, protocol, read, require_frozen, sha, verify_file, verify_prepared, write


def lock_identity(folder,identity):
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'identity.json'
    if path.exists():
        assert read(path) == identity, 'Job identity changed'
    else:
        write(path,identity)


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def construct(root,cohort,software_one_update=False):
    if software_one_update:
        from authority import is_software_fixture
        assert is_software_fixture(root)
    import torch
    from torch import nn
    from timm.data.mixup import Mixup
    import runtime as rt
    p = protocol()
    c = p['cohorts'][cohort]
    cfg = {'cohort':cohort,'architecture':c['architecture'],'seed':c['construction_seed'],
           'construction_seed':c['construction_seed'],'epochs':c['construction_epochs'],
           'dataset':'EuroSAT_RGB','input_size':64,'num_classes':10,'T':4,'batch':c['construction_batch'],
           'accumulation':c['construction_accumulation'],'precision':'AMP float16'}
    out = root/'construction'/cohort
    identity = {'config':cfg,'protocol_sha256':sha(BUNDLE/'protocol.json')}
    lock_identity(out,identity)
    if (out/'complete.json').exists() and not software_one_update:
        done = read(out/'complete.json')
        verify_file(out/'best.pt',done['checkpoint_sha256'])
        return
    rt.seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = rt.build_model(root,c['architecture'],4,cfg['seed'])
    initial_fingerprint = state_fingerprint(model)
    if cohort == 'C-conf':
        from timm.optim import create_optimizer_v2
        from timm.scheduler import create_scheduler_v2
        optimizer = create_optimizer_v2(model,opt='adamw',lr=c['construction_lr'],weight_decay=.01)
        scheduler,_ = create_scheduler_v2(optimizer,sched='cosine',num_epochs=cfg['epochs'],
             cooldown_epochs=10,min_lr=1e-5,warmup_lr=1e-5,warmup_epochs=3)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr=c['construction_lr'],weight_decay=.01)
        scheduler = None
    scaler = torch.amp.GradScaler('cuda')
    best,start,elapsed = -1.,0,0.
    if (out/'last.pt').exists():
        cp = torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert cp['identity'] == identity
        if c['architecture'] == 'resformer':
            model._revision_rv.v.load_preserving_buffers(model,cp['model'])
        else:
            model.load_state_dict(cp['model'],strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        scaler.load_state_dict(cp['scaler'])
        if scheduler:
            scheduler.load_state_dict(cp['scheduler'])
        start,best,elapsed = cp['epoch']+1,cp['best'],cp['process_seconds']
        del cp
    # Post-hook counts actual AdamW commits; GradScaler skips do not call optimizer.step.
    counters={'committed':max((int(v.get('step',0)) for v in optimizer.state.values()),default=0),'attempted':start*(18816//128)}
    def committed_hook(opt,args,kwargs):counters['committed']+=1
    optimizer.register_step_post_hook(committed_hook)
    software_before={n:p.detach().cpu().clone() for n,p in model.named_parameters()} if software_one_update else None
    software_attempts=[]
    log = out/'epochs.jsonl'
    if log.exists():
        kept = [line for line in log.read_text().splitlines() if json.loads(line)['epoch'] <= start]
        log.write_text('\n'.join(kept)+('\n' if kept else ''))
    mixup = Mixup(mixup_alpha=.8,cutmix_alpha=1.,prob=1.,switch_prob=.5,mode='batch',
                  label_smoothing=.1,num_classes=10)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,cfg['epochs']):
        rt.seed_all(cfg['seed']+epoch*100003)
        train,_ = rt.construction_loaders(root,cohort,epoch)
        if scheduler is None:
            rate = c['construction_lr']*(.01+.99*.5*(1+math.cos(math.pi*epoch/max(1,cfg['epochs']-1))))
            if epoch < 5:
                rate = c['construction_lr']*(epoch+1)/5
            for group in optimizer.param_groups:
                group['lr'] = rate
        rt.prepare_mode(model,True)
        tick = time.perf_counter()
        total,loss_total = 0,0.
        epoch_committed_start=counters['committed'];epoch_attempted_start=counters['attempted']
        optimizer.zero_grad(set_to_none=True)
        for index,(x,y) in enumerate(train):
            x,y = x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            x,target = mixup(x,y)
            with torch.autocast('cuda',dtype=torch.float16):
                z = rt.forward(model,x)
                loss = -(target*nn.functional.log_softmax(z,-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite construction loss: {cohort}, epoch {epoch+1}')
            group_start = (index//cfg['accumulation'])*cfg['accumulation']
            group_count = min(cfg['accumulation'],len(train)-group_start)
            scaler.scale(loss/group_count).backward()
            if (index+1)%cfg['accumulation']==0 or index+1==len(train):
                if cohort != 'C-conf':
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                before_commit=counters['committed'];before_scale=float(scaler.get_scale());counters['attempted']+=1
                scaler.step(optimizer)
                scaler.update()
                committed=counters['committed']-before_commit
                if software_one_update:
                    finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                    clocks=sorted(set(int(v['step']) for v in optimizer.state.values() if 'step' in v))
                    delta=sum(float((p.detach().cpu().double()-software_before[n].double()).square().sum()) for n,p in model.named_parameters())**.5
                    software_attempts.append({'attempt':counters['attempted'],'actual_commit':committed,'scale_before':before_scale,'scale_after':float(scaler.get_scale()),'finite_unscaled_gradients':finite,'populated_optimizer_clocks':clocks,'parameter_delta_L2_from_initial':delta})
                    if committed:
                        assert committed==1 and finite and clocks==[1] and delta>0,'actual nonzero finite AMP commit required'
                        rt.reset(model);torch.cuda.synchronize();write(root/'environment/construction_software_acceptance.json',{'status':'passed','software_fixture_only':True,'optimizer_steps':counters['committed'],'optimizer_attempted_steps':counters['attempted'],'attempts':software_attempts,'effective_batch':128,'microbatch':cfg['batch'],'accumulation':cfg['accumulation'],'configured_epochs':600,'executed_microbatches':index+1,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-tick});return
                    if len(software_attempts)>=8:raise RuntimeError('no actual AMPcommit within8 bounded syntheticattempts')
                optimizer.zero_grad(set_to_none=True)
            total += len(y)
            loss_total += float(loss.detach())*len(y)
            rt.reset(model)
        del train
        rate = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(epoch+1)
        validation = rt.evaluate_validation(root,model)
        seconds = time.perf_counter()-tick
        elapsed += seconds
        record = {'optimizer_attempted_steps_total':counters['attempted'],'optimizer_committed_steps_total':counters['committed'],'optimizer_skipped_steps_total':counters['attempted']-counters['committed'],'epoch_optimizer_attempts':counters['attempted']-epoch_attempted_start,'epoch_optimizer_commits':counters['committed']-epoch_committed_start,'epoch':epoch+1,'learning_rate':rate,'train_ce_soft_targets':loss_total/total,
                  'validation':validation,'epoch_process_seconds':seconds,'construction_process_seconds':elapsed}
        if validation['accuracy'] > best:
            best = validation['accuracy']
            rt.atomic_save({'model':model.state_dict(),'config':cfg,'epoch':epoch,
                            'model_metadata':model._revision_metadata},out/'best.pt')
        rt.atomic_save({'model':model.state_dict(),'config':cfg,'identity':identity,'epoch':epoch,
                        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                        'scheduler':scheduler.state_dict() if scheduler else None,
                        'best':best,'process_seconds':elapsed},out/'last.pt')
        with log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        write(out/'progress.json',record)
        print(json.dumps({'cohort':cohort,**record}),flush=True)
    differences = {'data_split':'official EuroSAT_RGB train11959/validation1712; logical128 drops84 at epoch boundary; test3421locked',
        'preprocessing':'EuroSAT_RGB training-only pixel mean/std; original research augmentation preserved',
        'identity_execution_wrappers':'historical identity-only wrappers retained on resformer; no added trainable parameters',
        'construction_loader':'epoch-specific auditable seeding; no claim of bitwise equality to original upstream training',
        'checkpoint_selection':'research held-out validation accuracy; never test accuracy',
        'physical_batch':cfg['batch'],'gradient_accumulation':cfg['accumulation'],'BN_and_mixup_not_equivalent_to_128':True,
        'construction_gradient_clipping':'none, matching official main.py' if cohort=='C-conf' else 'norm 1, matching research recipe',
        'official_scheduler_used':cohort=='C-conf','convergence_claim':'requires examination of actual curves; 600 epochs alone is not evidence'}
    differences['upstream_source_integrity']='reference YAML/main at official commit41408b4ef306bb932bd0db4c111136d930ee2e57 verified by Git blob SHA1; legacy identity model wrappers and research data pipeline remain distinct'
    write(out/'configuration_differences.json',differences)
    write(out/'complete.json',{'status':'complete','cohort':cohort,'config':cfg,
        'checkpoint':str(out/'best.pt'),'checkpoint_sha256':sha(out/'best.pt'),
        'random_initial_state_sha256':initial_fingerprint,'best_validation_accuracy':best,
        'optimizer_attempted_steps':counters['attempted'],'optimizer_committed_steps':counters['committed'],'optimizer_skipped_steps':counters['attempted']-counters['committed'],'construction_process_seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'configuration_differences':differences})


def adapt(root,job,stop_after=None):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from timm.data.mixup import Mixup
    import runtime as rt
    p = protocol()
    initial = Path(job['initial_checkpoint'])
    verify_file(initial,job['initial_checkpoint_sha256'])
    out,dest = root/'training'/job['id'],root/'models'/job['id']
    identity = {'job':job,'protocol_sha256':sha(BUNDLE/'protocol.json')}
    if stop_after is not None:
        assert job.get('software_fixture') is True and 0 < stop_after < job['steps']
    lock_identity(out,identity)
    dest.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        done = read(out/'complete.json')
        assert done['identity'] == identity
        for cp in done['checkpoints']:
            verify_file(cp['checkpoint'],cp['checkpoint_sha256'])
        return
    rt.configure()
    model = rt.load_model(root,job['architecture'],initial,job.get('T') or 4)
    rt.prepare_mode(model,True)
    for param in model.parameters():
        param.requires_grad_(True)
    parameters = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=job['lr'],weight_decay=.01)
    adapter = None
    if job['method']=='mtt':
        from mtt_adapter import StageAdapter
        adapter = StageAdapter(model)
    rt.seed_all(job['seed'])
    temporal_rng = np.random.default_rng(job['seed']+6_000_000)
    start,elapsed,evaluations = 0,0.,[]
    resume = dest/'resume.pt'
    if resume.exists():
        cp = torch.load(resume,map_location='cpu',weights_only=False)
        assert cp['identity'] == identity
        if job['architecture']=='resformer':
            model._revision_rv.v.load_preserving_buffers(model,cp['model'])
        else:
            model.load_state_dict(cp['model'],strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        start,elapsed,evaluations = cp['step'],cp['training_seconds'],cp['evaluations']
        rt.restore_rng(cp['rng'])
        temporal_rng.bit_generator.state = cp['temporal_rng']
        del cp
    data = rt.planned_dataset(root,job['seed'],job['steps'],start)
    loader = DataLoader(data,batch_size=p['adaptation']['microbatch'],shuffle=False,num_workers=6,pin_memory=True,
        persistent_workers=True,generator=torch.Generator().manual_seed(job['seed']+3_000_000))
    np.savez_compressed(out/'sample_plan.npz',image_id=data.indices,augmentation_seed=data.aug_seeds)
    mixup = Mixup(mixup_alpha=.8,cutmix_alpha=1.,prob=1.,switch_prob=.5,mode='batch',label_smoothing=.1,num_classes=10)
    initial_config = torch.load(initial,map_location='cpu',weights_only=False).get('config',{})
    logs = out/'steps.jsonl'
    if logs.exists():
        old = [s for s in logs.read_text().splitlines() if json.loads(s)['step']<=start]
        logs.write_text('\n'.join(old)+('\n' if old else ''))
    rt.prepare_mode(model,True)
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    for step in range(start+1,job['steps']+1):
        patterns = temporal_rng.integers(2,5,size=(3,3)).tolist() if adapter else [None]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for micro in range(128//p['adaptation']['microbatch']):
            x,y = next(iterator)
            assert len(y)==p['adaptation']['microbatch']
            x,y = x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            old_np = np.random.get_state()
            np.random.seed((job['seed']*100003+step*1009+micro)%(2**32-1))
            x,target = mixup(x,y)
            np.random.set_state(old_np)
            for pattern in patterns:
                if adapter:
                    adapter.set(pattern)
                z = rt.forward(model,x)
                loss = -(target*nn.functional.log_softmax(z,-1)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite adaptation loss at step {step}')
                (loss*(len(y)/128)).backward()
                loss_sum += float(loss.detach())*(len(y)/128)
                rt.reset(model)
        norm = nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-tick
        elapsed += seconds
        record = {'step':step,'loss_sum_configurations':loss_sum,'gradient_norm':float(norm),
                  'seconds':seconds,'training_seconds':elapsed,'unique_images':128,
                  'configuration_image_presentations':128*len(patterns),'patterns':patterns}
        with logs.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        if step in job['budgets'] or step==job['steps']:
            if adapter:
                adapter.set(None)
            checkpoint = {'model':model.state_dict(),'config':job,'initial_config':initial_config,
                          'step':step,'model_metadata':model._revision_metadata}
            rt.atomic_save(checkpoint,dest/f'step{step}.pt')
            before = rt.rng_state()
            values = {}
            for e in ((2,4) if adapter else (job['T'],)):
                rt.set_window(model,e)
                values[str(e)] = rt.evaluate_validation(root,model)
            evaluations.append({'step':step,'raw_validation_by_window':values,'training_seconds':elapsed})
            rt.restore_rng(before)
            rt.prepare_mode(model,True)
            if not adapter:
                rt.set_window(model,job['T'])
            write(out/'evaluations.json',evaluations)
        if step%256==0 or step==job['steps'] or step==stop_after:
            rt.atomic_save({'model':model.state_dict(),'identity':identity,'optimizer':optimizer.state_dict(),
                'step':step,'rng':rt.rng_state(),'temporal_rng':temporal_rng.bit_generator.state,
                'training_seconds':elapsed,'evaluations':evaluations},resume)
        if step%64==0 or step==job['steps']:
            write(out/'progress.json',record)
            print(json.dumps({'id':job['id'],**record}),flush=True)
        if step==stop_after:
            print('Stopped software-only trajectory at requested recovery boundary',flush=True)
            return
    if adapter:
        adapter.close()
    checkpoints = [{'step':k,'checkpoint':str(dest/f'step{k}.pt'),'checkpoint_sha256':sha(dest/f'step{k}.pt')}
                   for k in job['budgets'] if k<=job['steps']]
    write(out/'complete.json',{'status':'complete','identity':identity,'step':job['steps'],
        'training_seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'evaluations':evaluations,'checkpoints':checkpoints})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--construct',choices=['C-conf','F-arch'])
    parser.add_argument('--job',type=Path)
    parser.add_argument('--stop-after',type=int)
    args = parser.parse_args()
    root = verify_prepared(args.root)
    environment(root)
    assert bool(args.construct) != bool(args.job)
    job = read(args.job) if args.job else None
    software = bool(job and job.get('software_fixture') is True and job.get('cohort')=='SOFTWARE' and job.get('id','').startswith('SOFTWARE_'))
    if not software:
        require_frozen(root)
        assert read(root/'environment/preflight.json')['status']=='passed'
        assert read(root/'environment/model_smoke.json')['status']=='passed'
    construct(root,args.construct) if args.construct else adapt(root,job,args.stop_after)
