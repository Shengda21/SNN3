"""Complete four-view fixed-S0 CE, transactional updates and quarter-tail means."""
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
from port_common import *

def train(args):
    import torch
    import ce,base,state
    from tail_average import TailAverage
    rt=runtime(args.bundle,args.construction_root)
    from normalization import apply_bank
    root=Path(args.construction_root).resolve();out=Path(args.out).resolve()
    require(not out.exists(),'preserve prior run and failure scene; use a fresh output')
    source=root/'construction/C-conf/best.pt';done=read(root/'construction/C-conf/complete.json')
    require(done['status']=='complete' and sha(source)==done['checkpoint_sha256'],'completed600epoch own S0')
    require(done['config']['dataset']=='CIFAR100' and done['config']['epochs']==600,'full CIFAR100 construction')
    role=read(root/'environment/construction_role.json')['role']
    allowed={'development':(23611,23612),'formal1':(23651,),'formal2':(23652,),'formal3':(23653,)}
    require(args.seed in allowed[role],'frozen paired stream for construction role')
    if role!='development' and not args.acceptance_only:
        require(args.development_gate is not None,'original complete development gate required')
        verify_development_gate(args.development_gate,root)
    guard_training();ce.R=root
    initial=torch.load(source,map_location='cpu',weights_only=False)
    native=rt.load_model(root,'resformer',source,2)
    op=ce.feature_module();probe=op.WindowBank(native)
    job={'id':out.name,'architecture':'resformer','cohort':'CIFAR100-Resformer','seed':args.seed,
         'initial_checkpoint':str(source),'initial_checkpoint_sha256':sha(source),
         'bn_count':len(probe.names),'bn_channels_total':sum(probe.channels),'adapter_parameter_count':probe.parameter_count}
    del native,probe
    prefix,shadow,bank,opts=ce.build(job,rt)
    rt.seed_all(args.seed)
    banks=ce.prepare_banks(base,prefix,shadow,bank,state,rt,apply_bank)
    out.mkdir(parents=True);rt.atomic_save(banks,out/'banks.pt')
    identity={'schema':SCHEMA,'job':job,'authority':verify_parent(args.bundle,root),'construction_role':read(root/'environment/construction_role.json'),
              'hardware':read(root/'environment/actual_hardware.json'),'previously_observed_formal_starts':False,
              'test_used':False,'validation_used_during_training':False}
    write(out/'started.json',identity)
    data=rt.planned_dataset(root,args.seed,2048)
    loader=torch.utils.data.DataLoader(data,batch_size=64,shuffle=False,num_workers=args.workers,
           pin_memory=True,persistent_workers=args.workers>0,generator=torch.Generator().manual_seed(args.seed+3000000))
    np.savez_compressed(out/'sample_plan.npz',image_id=data.indices,augmentation_seed=data.aug_seeds)
    iterator=iter(loader);tail=TailAverage(dict(bank.named_parameters()));tail.last_step=0
    S0=base.native_state(prefix,cpu=True);elapsed=0.;records=[]
    def mode_bridge(payload,batches,label):
        from deploy import load
        path=out/(label+'_scratch.pt');payload.update(schema=SCHEMA,temperature_pending=True);rt.atomic_save(payload,path)
        bridges=[];saved=state.state_snapshot(shadow)
        for t in (2,4):
            for policy in ('raw','matched'):
                bn=banks['banks'][str(t)] if policy=='matched' else base.persistent_bank(prefix)
                state.restore_state(shadow,saved,rt);apply_bank(shadow,bn);rt.set_window(shadow,t);state.set_locked_mode(shadow,rt)
                with torch.no_grad(),op.installed(shadow,bank):a=rt.forward(shadow,batches[0][0]).detach().cpu()
                model=load(root,path,t,rt,scale=False);apply_bank(model,bn);rt.prepare_mode(model,False)
                with torch.no_grad():b=rt.forward(model,batches[0][0]).detach().cpu()
                require(ce.equal_tree(a,b),label+' four-view locked/native deployment bridge')
                bridges.append({'window':t,'BN':policy,'bytes_exact':True});del model
        state.restore_state(shadow,saved,rt);return bridges
    try:
        count=2 if args.acceptance_only else 2048
        for step in range(1,count+1):
            batches=base.cuda_batches(iterator);torch.cuda.synchronize();tick=time.perf_counter()
            if args.acceptance_only and step==1:
                payload=ce.pack(prefix,bank,initial,job,0,opts,banks,copy.deepcopy(rt.rng_state()),identity)
                zero_bridges=mode_bridge(payload,batches,'zero')
            r=ce.one_attempt(base,prefix,shadow,bank,opts,batches,state,rt,apply_bank,banks,step)
            averaged=tail.committed(step,bank.state_dict());torch.cuda.synchronize();elapsed+=time.perf_counter()-tick
            record={'step':step,'training_seconds':elapsed,'native_S0_unchanged':r['transport']['native_S0_unchanged'],
                    'groups':{g:{k:r[g][k] for k in ('objective','gradient_norm','nonzero_update','parameter_delta_L2','optimizer_steps')} for g in ce.GROUPS}}
            with (out/'steps.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if step in BUDGETS:
                require(averaged is not None and ce.equal_tree(S0,base.native_state(prefix,cpu=True)),'S0 retention at budget')
                p=ce.pack(prefix,bank,initial,job,step,opts,banks,copy.deepcopy(rt.rng_state()),identity)
                rt.atomic_save(p,out/f'instant_step{step}.pt')
                values,receipt=averaged;p.update(schema=SCHEMA,compensation=ce.cpu_tree(values),average=receipt,temperature_pending=True)
                path=out/f'mean_step{step}.pt';rt.atomic_save(p,path)
                rt.atomic_save({'sum_fp64':tail.last_sum,'average':receipt,'parameter_names':list(tail.names)},out/f'mean_evidence_step{step}.pt')
                records.append({'step':step,'mean':str(path),'mean_sha256':sha(path)})
            if step==1 or step%16==0:print(json.dumps(record),flush=True)
        require(ce.equal_tree(S0,base.native_state(prefix,cpu=True)),'final S0 bytes')
        if args.acceptance_only:
            # Populated transaction rollback exercises all optimizer states.
            maps=ce.cpu_tree(bank.state_dict());moments=ce.cpu_tree({k:o.state_dict() for k,o in opts.items()})
            try:
                ce.one_attempt(base,prefix,shadow,bank,opts,batches,state,rt,apply_bank,banks,3,force_exception='after_t4_matched')
            except RuntimeError as exc:
                require(str(exc)=='intentional_after_t4_matched','expected exact rollback injection')
            else:raise RuntimeError('rollback injection did not fail')
            require(ce.equal_tree(maps,ce.cpu_tree(bank.state_dict())) and
                    ce.equal_tree(moments,ce.cpu_tree({k:o.state_dict() for k,o in opts.items()})),'populated exact rollback')
            scratch=ce.pack(prefix,bank,initial,job,2,opts,banks,copy.deepcopy(rt.rng_state()),identity)
            bridges=mode_bridge(scratch,batches,'nonzero')
            write(out/'acceptance.json',{'status':'passed','populated_optimizer_transaction_rollback':True,
                  'zero_four_view_native_deployment_bridge':zero_bridges,'nonzero_four_view_native_deployment_bridge':bridges,
                  'test_read':False,'validation_read':False})
        write(out/'complete.json',{'status':'acceptance_only' if args.acceptance_only else 'complete',
              'identity':identity,'attempts':count,'optimizer_calls':4*count,'training_TIS':1536*count,
              'initial_calibration_TIS':12288,'calibrations':2,'training_seconds':elapsed,
              'checkpoints':records,'sample_plan_sha256':sha(out/'sample_plan.npz'),'test_used':False})
    except BaseException:
        write(out/'failed.json',{'status':'failed_preserved','automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--construction-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--acceptance-only',action='store_true');p.add_argument('--development-gate',type=Path);train(p.parse_args())
