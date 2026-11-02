import sys as _s
from pathlib import Path as _P
_s.path.insert(0,str(_P(__file__).resolve().parent.parent))
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
    from authority import verify_sources
    import dataset
    root,data=Path(root).resolve(),Path(data).resolve();verify_sources()
    assert root!=data and root not in data.parents
    marker=root/'environment/eurosat64_prepared.json'
    if marker.exists():
        r=read(marker);assert r['protocol_sha256']==sha(B/'protocol.json') and r['data_receipt_sha256']==sha(data/'training_data_receipt.json');return root
    assert not root.exists() or not any(root.iterdir()),'fresh independent root required'
    x,y=dataset.read_arrays(data,'train');assert len(y)==18900
    for n in ('environment','state','construction','training','models','scoring','analysis','legacy/code'):(root/n).mkdir(parents=True,exist_ok=True)
    vision=root/'archive/direction1_budget_rotation_20260907';shutil.copytree(B/'payload/vision',vision);shutil.copytree(B/'payload/legacy_code',root/'legacy/code',dirs_exist_ok=True)
    (vision/'data').mkdir(exist_ok=True);write(vision/'data/EuroSAT_RGB_split.json',dataset.split());write(root/'environment/dataset.json',{'data_path':str(data)})
    sources=[q for q in B.rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.suffix!='.pyc']
    write(marker,{'protocol_sha256':sha(B/'protocol.json'),'source_freeze_sha256':verify_sources(),'train_samples':18900,'split_sha256':sha(vision/'data/EuroSAT_RGB_split.json'),'data_receipt_sha256':sha(data/'training_data_receipt.json'),'sources':[{'path':q.relative_to(B).as_posix(),'sha256':sha(q)} for q in sources]})
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
    from authority import require_production
    require_production()
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
