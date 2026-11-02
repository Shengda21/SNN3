"""Full-model GPU software checks; none of these states belong to scientific cohorts."""
import argparse
import gc
from pathlib import Path
import subprocess
import sys
from common import BUNDLE, environment, read, sha, verify_prepared, write


def main(root):
    root = verify_prepared(root)
    environment(root)
    assert read(root/'environment/preflight.json')['status']=='passed'
    import torch
    import runtime as rt
    rt.configure()
    reports = []
    for architecture in ('resformer','spikformer'):
        folder = root/'software_checks'/architecture
        folder.mkdir(parents=True,exist_ok=True)
        initial = folder/'random_initial.pt'
        if not initial.exists():
            model = rt.build_model(root,architecture,4,15998)
            rt.atomic_save({'model':model.state_dict(),'config':{'dataset':'CIFAR100','seed':15998,
                'architecture':architecture},'software_fixture':True},initial)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        def run(name,stop=None,T=4,method='native'):
            job = {'id':f'SOFTWARE_{architecture}_{name}','cohort':'SOFTWARE','architecture':architecture,
                'method':method,'T':T,'seed':15999,'lr':2e-5,'steps':4,'budgets':[2,4],
                'initial_checkpoint':str(initial),'initial_checkpoint_sha256':sha(initial),
                'roles':['software_only'],'software_fixture':True}
            path = folder/f'{name}.json'
            write(path,job)
            command = [sys.executable,str(BUNDLE/'scripts/train_job.py'),'--root',str(root),'--job',str(path)]
            if stop:
                command += ['--stop-after',str(stop)]
            subprocess.run(command,check=True)
            return root/'models'/job['id']/'resume.pt'
        a = run('continuous')
        run('resumed',2)
        b = run('resumed')
        left = torch.load(a,map_location='cpu',weights_only=False)
        right = torch.load(b,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,right['model'][k]) and v.dtype==right['model'][k].dtype for k,v in left['model'].items())
        assert all(torch.equal(v,right['optimizer']['state'][k][n]) if torch.is_tensor(v) else v==right['optimizer']['state'][k][n]
                   for k,state in left['optimizer']['state'].items() for n,v in state.items())
        run('two_step',T=2)
        model = rt.load_model(root,architecture,initial,4)
        banks = []
        for window in (2,4):
            rt.set_window(model,window)
            _,report = rt.calibrate(root,model,9640)
            assert report['protected_values_and_dtypes_unchanged']
            banks.append({'T':window,**report})
        if architecture=='resformer':
            # Test the actual intended physical C batch, rather than equating accumulation with BN.
            images,ids = rt.calibration_images(root,9640)
            rt.prepare_mode(model,True)
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                output = rt.forward(model,images[:128].cuda())
                loss = output.square().mean()
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.zero_grad(set_to_none=True)
            rt.reset(model)
            del images,output,loss
        del model,left,right
        gc.collect()
        torch.cuda.empty_cache()
        if architecture=='resformer':
            run('mtt',T=None,method='mtt')
        reports.append({'architecture':architecture,'continuous_resume_model_and_optimizer_bitwise_equal':True,
            'native_T2_T4_forward_backward_finite':True,'BN_checks':banks,
            'C_physical_128_AMP_backward_passed':architecture=='resformer',
            'MTT_stagewise_forward_backward_passed':architecture=='resformer'})
    write(root/'environment/model_smoke.json',{'status':'passed','checks':reports,
        'scope':'GPU software functional acceptance only; no research results',
        'test_split_used':False,'scientific_cohort_states_created':False})
    print('Full-model software acceptance passed; no formal cohort trained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
