"""Parallel byte-range transfer from torchvision's official CIFAR100 source."""
import concurrent.futures,hashlib,os,subprocess,sys,time
from pathlib import Path
import requests
from torchvision.datasets import CIFAR100

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'direction1_budget_rotation_20260907/data'
DATA.mkdir(parents=True,exist_ok=True)
TARGET=DATA/CIFAR100.filename
MIRROR='https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-100-python.tar.gz'

def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def fetch(part):
    i,start,end=part;p=DATA/f'cifar_chunk_{i}.partial'
    for attempt in range(4):
        try:
            with requests.get(CIFAR100.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                with p.open('wb') as f:
                    for b in r.iter_content(65536):f.write(b)
            assert p.stat().st_size==end-start+1
            print(f'chunk {i} complete',flush=True);return p
        except Exception:
            if attempt==3:raise
            time.sleep(2)

if not TARGET.exists() or md5(TARGET)!=CIFAR100.tgz_md5:
    import json
    temp=TARGET.with_suffix('.domestic.partial');started=time.time()
    for attempt in range(3):
        try:
            with requests.get(MIRROR,stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                with temp.open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
            assert md5(temp)==CIFAR100.tgz_md5,'Official CIFAR100 MD5 mismatch'
            sha=hashlib.sha256(temp.read_bytes()).hexdigest()
            assert sha=='85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
            temp.replace(TARGET)
            break
        except Exception:
            if attempt==2:raise
            time.sleep(3)
    receipt={'mirror':MIRROR,'official_source':CIFAR100.url,'sha256':sha,'size':TARGET.stat().st_size,'seconds':time.time()-started}
    (ROOT/'results/paper_experiments_20260908/E0/visual_download.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
subprocess.run([sys.executable,str(ROOT/'code/paper_vision_restore.py'),'--prepare-only'],check=True)
