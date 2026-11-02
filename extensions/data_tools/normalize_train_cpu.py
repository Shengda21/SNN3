"""Train-only float64 moments after a declared deterministic 64px evaluation view."""
import argparse,concurrent.futures,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

def moments(path):
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        image=ImageOps.fit(image,(64,64),method=Image.Resampling.LANCZOS,centering=(.5,.5))
        a=np.asarray(image,dtype=np.float64)/255.
        return a.sum((0,1)),(a*a).sum((0,1)),64*64

def run(args):
    manifest=Path(args.manifest);raw=manifest.read_bytes();document=json.loads(raw)
    output=manifest.parent/'normalization_train_only_draft.json'
    assert not output.exists(),'Preserve completed train normalization'
    train=[r for r in document['records'] if r['split']=='train']
    held={r['global_id'] for r in document['records'] if r['split']!='train'}
    assert not held.intersection(r['global_id'] for r in train)
    total=np.zeros(3,dtype=np.float64);square=np.zeros(3,dtype=np.float64);count=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for x,x2,n in pool.map(moments,(Path(args.root)/r['path'] for r in train)):
            total+=x;square+=x2;count+=n
    mean=total/count;std=np.sqrt(square/count-mean*mean);assert np.all(std>0)
    result={'status':'deterministic_data_reconstruction','samples':len(train),'pixels_per_channel':count,
            'mean':mean.tolist(),'std':std.tolist(),'dtype':'float64 population moments after RGB/255',
            'evaluation_view':'EXIF transpose, RGB, aspect-preserving center ImageOps.fit64 LANCZOS; native64 unchanged',
            'manifest_sha256':hashlib.sha256(raw).hexdigest(),'training_ID_sha256':hashlib.sha256(json.dumps([r['global_id'] for r in train]).encode()).hexdigest(),
            'validation_images_read':False,'test_images_read':False,'GPU_used':False}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);run(p.parse_args())
