"""Run exactly one frozen CIFAR-10 Resformer 600-epoch construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import platform
import subprocess

B = Path(__file__).resolve().parent
sys.path.insert(0,str(B/'scripts'))
from common import read, write, sha

def prepare(root,data):
    import numpy as np
    root,data=Path(root).resolve(),Path(data).resolve()
    if root == data or root in data.parents:
        raise ValueError('Data directory must be outside the execution root')
    marker=root/'environment/cifar100_prepared.json'
    if marker.exists():
        receipt=read(marker)
        assert receipt['protocol_sha256']==sha(B/'protocol.json')
        for row in receipt['sources']:
            assert sha(B/row['path'])==row['sha256']
        assert sha(root/'archive/direction1_budget_rotation_20260907/data/CIFAR100_split.json') == receipt['split_sha256']
        return root
    if root.exists() and any(root.iterdir()):
        raise ValueError('New independent campaign requires an empty execution root')
    # CIFAR100 constructor integrity-checks the retained Python batch files. No test
    # dataset is instantiated and no test labels are selected or scored here.
    from torchvision.datasets import CIFAR100
    ds=CIFAR100(str(data),train=True,download=False)
    assert len(ds)==50000 and set(ds.targets)==set(range(100))
    for name in ('environment','state','construction','training','models','scoring','analysis','legacy/code'):
        (root/name).mkdir(parents=True,exist_ok=True)
    vision=root/'archive/direction1_budget_rotation_20260907'
    shutil.copytree(B/'payload/vision',vision)
    shutil.copytree(B/'payload/legacy_code',root/'legacy/code',dirs_exist_ok=True)
    (vision/'data').mkdir(exist_ok=True)
    # Keep a single data copy, with an explicit absolute link inside this new root.
    (vision/'data/cifar-100-python').symlink_to(data/'cifar-100-python',target_is_directory=True)
    ids=np.random.default_rng(20260907).permutation(50000)
    split={'dataset':'CIFAR100','seed':20260907,'train':ids[:45000].tolist(),
           'validation':ids[45000:].tolist(),'test_used_for_selection':False}
    write(vision/'data/CIFAR100_split.json',split)
    sources=[x for x in B.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix not in ('.pyc',)]
    write(marker,{'protocol_sha256':sha(B/'protocol.json'),'train_samples':50000,
                  'split_sha256':sha(vision/'data/CIFAR100_split.json'),
                  'sources':[{'path':x.relative_to(B).as_posix(),'sha256':sha(x)} for x in sources]})
    return root

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--role',choices=['development','formal1','formal2','formal3'],required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    root=prepare(args.root,args.data)
    role_path=root/'environment/construction_role.json'
    role_identity={'role':args.role,'seed':read(B/'protocol.json')['construction_roles'][args.role],
          'protocol_sha256':sha(B/'protocol.json'),'execution_scope':'reader_replication'}
    if role_path.exists():
        assert read(role_path)==role_identity, 'Construction role changed: use a fresh root'
    else:
        write(role_path,role_identity)
    if args.prepare_only:
        return
    import common
    import runtime
    import train_job
    import torch
    hardware={'hostname':platform.node(),'platform':platform.platform(),'python':sys.version,
              'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
              'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version',
                          '--format=csv,noheader'],capture_output=True,text=True,check=True)
    hardware['nvidia_smi']=query.stdout.strip().splitlines()
    write(root/'environment/actual_hardware.json',hardware)
    p=read(B/'protocol.json')
    p['cohorts']['C-conf']['construction_seed']=p['construction_roles'][args.role]
    # Each role gets its own execution root; no old checkpoint can be imported.
    common.protocol=lambda:p
    runtime.protocol=lambda:p
    train_job.protocol=lambda:p
    train_job.construct(root,'C-conf')

if __name__=='__main__':
    main()
