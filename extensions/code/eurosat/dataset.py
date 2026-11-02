"""Pinned RGB64 source mapping; full-method heldout authority, no caller bypass."""
import json,hashlib
from pathlib import Path
import numpy as np
from authority import B,read,sha,require_production,is_software_fixture
_VALIDATION_OPEN=False
_IMAGE_PHASES={}
def specification():return read(B/'constructor/protocol.json')['data']
def split():
 d=specification();off=0;result={'dataset':d['dataset']}
 for k in ('train','validation','test'):result[k]=list(range(off,off+d[k]));off+=d[k]
 return result

def read_arrays(data,phase,root=None,test_context=None):
 global _VALIDATION_OPEN,_IMAGE_PHASES
 data=Path(data);receipt=read(data/'training_data_receipt.json');d=specification();prior=_VALIDATION_OPEN
 # The whole val access, including SHA opens, is inside the same scope.
 _VALIDATION_OPEN=phase=='validation'
 try:
  if receipt.get('software_fixture'):
   assert root is not None and is_software_fixture(root)
   files=[phase+'_images.npy',phase+'_labels.npy']
   for n in files:assert sha(data/n)==receipt['files'][n]
   return np.load(data/files[0],mmap_mode='r'),np.load(data/files[1],mmap_mode='r')
  if phase in ('validation','test'):require_production()
  elif phase!='train':raise ValueError(phase)
  if phase=='test':
   assert root is not None and test_context is not None
   import sys;sys.path.insert(0,str(B/'method'));from port_common import verify_test_freeze
   verify_test_freeze(test_context['freeze'],test_context['checkpoint'],root,test_context['method'])
  p=read(B/'constructor/protocol.json');expected=p['normalization'];assert sha(data/'training_data_receipt.json')==p['training_receipt_sha256'];assert receipt['files']==expected['files'] and receipt['source_manifest_sha256']==p['data_manifest_sha256']
  assert sha(data/'source_records.json')==receipt['files']['source_records.json'];records=read(data/'source_records.json')['records'];source=Path(receipt['source_root']);_IMAGE_PHASES={str((source/r['path']).resolve()):r['split'] for r in records}
  if phase=='train':
   for n in ('train_images.npy','train_labels.npy'):assert sha(data/n)==receipt['files'][n]
   x=np.load(data/'train_images.npy',mmap_mode='r');y=np.load(data/'train_labels.npy',mmap_mode='r')
  else:
   from PIL import Image,ImageOps
   rows=[r for r in records if r['split']==phase];images=[]
   for r in rows:
    q=source/r['path'];assert q.resolve().is_relative_to(source.resolve()) and sha(q)==r['file_sha256']
    with Image.open(q) as im:
     im=ImageOps.exif_transpose(im).convert('RGB')
     if p['data']['native64']:assert im.size==(64,64)
     else:im=im.resize((64,64),Image.Resampling.BILINEAR)
     images.append(np.asarray(im,dtype=np.uint8))
   x=np.stack(images);y=np.asarray([r['label'] for r in rows],dtype=np.int64)
  assert x.shape==(d[phase],64,64,3) and x.dtype==np.uint8 and y.shape==(d[phase],) and y.dtype==np.int64 and ((y>=0)&(y<d['classes'])).all()
  return x,y
 finally:_VALIDATION_OPEN=prior
class Images:
 def __init__(self,data,phase,transform=None,root=None,test_context=None):self.x,self.y=read_arrays(data,phase,root,test_context);self.transform=transform
 def __len__(self):return len(self.y)
 def __getitem__(self,n):
  from PIL import Image
  im=Image.fromarray(self.x[n]);return (self.transform(im) if self.transform else im),int(self.y[n])
class LogicalEpoch:
 def __init__(self,seed,epoch,microbatch):assert 128%microbatch==0;self.seed=seed;self.epoch=epoch;self.microbatch=microbatch
 def __iter__(self):
  n=specification()['train'];usable=n//128*128;ids=np.random.default_rng(self.seed+self.epoch*100003).permutation(n)[:usable]
  for i in range(0,usable,self.microbatch):yield ids[i:i+self.microbatch].tolist()
 def __len__(self):return specification()['train']//128*128//self.microbatch
