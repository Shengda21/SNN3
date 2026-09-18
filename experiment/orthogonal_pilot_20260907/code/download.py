import json, time
from pathlib import Path
import requests

root = Path(__file__).resolve().parents[1]
url = 'https://modelscope.cn/models/kailai1104/SmoothSpike/resolve/master/checkpoints/smoothspike-bert-base/model.safetensors'
target = root/'data/model.safetensors'
if not target.exists():
    response = requests.get(url, stream=True, timeout=(30,120))
    response.raise_for_status()
    count = 0; started = time.time()
    with target.with_suffix('.partial').open('wb') as f:
        for chunk in response.iter_content(8*1024*1024):
            f.write(chunk); count += len(chunk)
            print(json.dumps({'download_mb':round(count/1e6,1),'elapsed_s':round(time.time()-started,1)}), flush=True)
    target.with_suffix('.partial').rename(target)
print('CHECKPOINT_READY', target.stat().st_size, flush=True)
for split in ['train','validation']:
    path=root/f'data/{split}.parquet'
    if not path.exists():
        data_url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
        response=requests.get(data_url,timeout=(20,120));response.raise_for_status()
        path.write_bytes(response.content)
    print('DATA_READY',split,path.stat().st_size,flush=True)
