"""Official split readers with disjoint global IDs and bound heldout access."""
import io,json,zipfile,hashlib,random
from pathlib import Path
import numpy as np
from authority import B,read,sha,require_production
COUNTS={'train':11959,'validation':1712,'test':3421};OFFSETS={'train':0,'validation':11959,'test':13671};PREFIX={'train':'train','validation':'val','test':'test'}
_VALIDATION_OPEN=False
EXPECTED='4865e31d7fc4142e6f9a49c52b407b512dbe202e4ab7dd3915f824988e6cb0e6'
def split():return {'dataset':'BloodMNIST64',**{k:list(range(v,v+COUNTS[k])) for k,v in OFFSETS.items()}}
def read_arrays(data,phase,root=None,test_context=None):
 global _VALIDATION_OPEN
 data=Path(data);receipt=read(data/'training_data_receipt.json')
 if receipt.get('software_fixture') is True:
  from authority import is_software_fixture
  assert root is not None and is_software_fixture(root)
  names=[phase+'_images.npy',phase+'_labels.npy']
  prior=_VALIDATION_OPEN;_VALIDATION_OPEN=(phase=='validation')
  try:
   for n in names:assert sha(data/n)==receipt['files'][n]
   return np.load(data/names[0],mmap_mode='r'),np.load(data/names[1],mmap_mode='r')
  finally:_VALIDATION_OPEN=prior
 if phase=='train':
  protocol=read(B/'constructor/protocol.json');frozen=protocol['normalization']
  assert receipt['archive_sha256']==EXPECTED and receipt['files']==frozen['files'] and receipt['mean']==frozen['mean'] and receipt['std']==frozen['std'],'official trainingauthority mismatch'
  for n,h in receipt['files'].items():assert sha(data/n)==h
  x=np.load(data/'train_images.npy',mmap_mode='r');y=np.load(data/'train_labels.npy',mmap_mode='r')
  assert x.shape==(11959,64,64,3) and x.dtype==np.uint8 and y.shape==(11959,) and y.dtype==np.int64 and ((y>=0)&(y<8)).all(),'official trainingarray contract'
  return x,y
 require_production()
 if phase=='test':
  assert root is not None and test_context is not None
  import sys;sys.path.insert(0,str(B/'method'));from port_common import verify_test_freeze
  verify_test_freeze(test_context['freeze'],test_context['checkpoint'],root,test_context['method'])
 elif phase!='validation':raise ValueError(phase)
 archive=Path(receipt['archive_path'])
 keys=[PREFIX[phase]+'_images',PREFIX[phase]+'_labels'];arrays=[]
 prior=_VALIDATION_OPEN;_VALIDATION_OPEN=(phase=='validation')
 try:
  assert sha(archive)==EXPECTED
  with zipfile.ZipFile(archive) as z:
   for key in keys:
    with z.open(key+'.npy') as f:arrays.append(np.load(io.BytesIO(f.read()),allow_pickle=False))
 finally:_VALIDATION_OPEN=prior
 x,y=arrays;assert x.shape==(COUNTS[phase],64,64,3) and x.dtype==np.uint8
 y=y.reshape(-1).astype(np.int64);assert y.shape==(COUNTS[phase],) and ((y>=0)&(y<8)).all()
 return x,y
class Images:
 def __init__(self,data,phase,transform=None,root=None,test_context=None):self.x,self.y=read_arrays(data,phase,root,test_context);self.transform=transform;self.offset=OFFSETS[phase]
 def __len__(self):return len(self.y)
 def __getitem__(self,n):
  from PIL import Image
  x=Image.fromarray(self.x[n]);return (self.transform(x) if self.transform else x),int(self.y[n])
class LogicalEpoch:
 def __init__(self,seed,epoch,microbatch):assert 128%microbatch==0;self.seed=seed;self.epoch=epoch;self.microbatch=microbatch
 def __iter__(self):
  ids=np.random.default_rng(self.seed+self.epoch*100003).permutation(11959)[:11904]
  for i in range(0,11904,self.microbatch):yield ids[i:i+self.microbatch].tolist()
 def __len__(self):return 11904//self.microbatch
