"""Original full-backbone T2/T4 or stagewise MTT baseline; fresh CIFAR100 jobs."""
import argparse,sys
from pathlib import Path
from port_common import *

def run(args):
    root=Path(args.construction_root).resolve();bundle=Path(args.bundle).resolve();out=Path(args.out).resolve()
    require(not out.exists(),'new baseline root required');out.mkdir(parents=True)
    (out/'legacy').symlink_to(root/'legacy',target_is_directory=True)
    (out/'archive').symlink_to(root/'archive',target_is_directory=True)
    job=read(args.job)
    require(job['steps']==2048 and job['budgets']==list(BUDGETS) and job['architecture']=='spikformer','fixed baseline budgets and architecture')
    require(job['method'] in ('native','mtt') and (job['method']=='mtt' or job['T'] in (2,4)),'full baseline method')
    role=read(root/'environment/construction_role.json')['role']
    allowed={'development':(23711,23712),'formal1':(23751,),'formal2':(23752,),'formal3':(23753,)}
    require(job['seed'] in allowed[role],'frozen paired role-seed')
    rates=read(bundle/'protocol.json')['mtt_rates' if job['method']=='mtt' else 'native_rates']
    require(job['lr'] in rates,'unchanged original baseline LR grid')
    require(sha(job['initial_checkpoint'])==job['initial_checkpoint_sha256'] and job['initial_checkpoint']==str(root/'construction/F-arch/best.pt'),'own S0 identity')
    rt=runtime(bundle,root)
    verify_checkpoint(root,job['initial_checkpoint'],'S0')
    if role!='development':
        pin=job['selection_authority'];require(sha(pin['path'])==pin['sha256'],'frozen actual development LR selection')
        selection=read(pin['path']);method='mtt' if job['method']=='mtt' else 'native'+str(job['T'])
        require(selection['schema']==SCHEMA and selection['source_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json')
                and selection['construction_root']==str(root.parent/'development') and selection['paired_seeds']==[23711,23712]
                and selection['choices'][method]['rate']==job['lr'] and selection['selection_uses_test'] is False,'actual selected formal rate')
        for evidence in selection['evidence']:require(sha(evidence['path'])==evidence['sha256'],'unchanged LR evidence')
        gate=job['development_gate_authority'];require(sha(gate['path'])==gate['sha256'],'frozen development gate artifact')
        verify_development_gate(gate['path'],root)
    job['campaign_authority']=verify_parent(bundle,root)
    guard_training()
    import train_job
    # Baseline outputs get an isolated tree while immutable raw sources remain in
    # the corresponding construction root. No old prepared identity is reused.
    original_rt=sys.modules['runtime'];original_rt.planned_dataset=rt.planned_dataset
    original_rt.validation_loader=rt.validation_loader
    original_rt.vision_root=lambda unused:root/'archive/direction1_budget_rotation_20260907'
    p=read(bundle/'protocol.json');train_job.protocol=lambda:p
    train_job.adapt(out,job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--construction-root',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
