import os,platform,subprocess
import torch
def actual():
 return {'hostname':platform.node(),'torch':torch.__version__,'cuda':torch.version.cuda,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True).splitlines()}
