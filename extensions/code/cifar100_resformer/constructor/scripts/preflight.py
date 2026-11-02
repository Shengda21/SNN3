"""Explicit server runtime check. No formal experiment is started."""
import argparse
from pathlib import Path
import shutil
import sys
from common import environment, protocol, verify_prepared, write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root = verify_prepared(parser.parse_args().root)
    environment(root)
    p = protocol()['runtime']
    import torch
    import torchvision
    import cupy
    assert sys.version_info[:2]==(3,12)
    assert torch.__version__.split('+')[0]==p['torch'] and torchvision.__version__.split('+')[0]==p['torchvision']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.version.cuda==p['cuda']
    assert torch.cuda.get_device_properties(0).total_memory/2**30>=p['minimum_GPU_GiB']
    assert shutil.disk_usage(root).free/2**30>=p['minimum_free_GiB']
    assert float(torch.ones(8,device='cuda').sum())==8
    assert int(cupy.arange(8).sum().get())==28
    write(root/'environment/preflight.json',{'status':'passed','device':torch.cuda.get_device_name(),
        'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,
        'scope':'runtime kernels only; full-model acceptance remains required'})
    print('Runtime passed; next run explicit smoke before scientific training')
