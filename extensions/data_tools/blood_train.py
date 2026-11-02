"""Official NPZ member isolation: train readers cannot request held-out arrays."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
EXPECTED_MD5='2b94928a2ae4916078ca51e05b6b800b';EXPECTED_BYTES=156251765
KEYS={'train':('train_images','train_labels'),'validation':('val_images','val_labels'),'test':('test_images','test_labels')}
COUNTS={'train':11959,'validation':1712,'test':3421}
OFFSETS={'train':0,'validation':11959,'test':13671}
def global_ids(split):return np.arange(OFFSETS[split],OFFSETS[split]+COUNTS[split],dtype=np.int64)
def local_id(split,global_id):
 i=int(global_id)-OFFSETS[split]
 if not 0<=i<COUNTS[split]:raise ValueError('global ID belongs to a different split')
 return i
def digest(path,kind='sha256'):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,kind).hexdigest()
class MemberReader:
 def __init__(self,path,phase='train',authorization=None,verify_official=True):
  self.path=Path(path);self.phase=phase
  if phase not in KEYS:raise ValueError('unknown phase')
  if phase!='train':raise PermissionError('this preparation reader opens training members only; heldout members are read by the frozen evaluation entrypoint')
  if verify_official:
   assert self.path.stat().st_size==EXPECTED_BYTES and digest(self.path,'md5')==EXPECTED_MD5,'official archive bytes changed'
  self._allowed=frozenset(KEYS[phase]);self.member_reads=[]
 def array(self,key):
  if key not in self._allowed:raise PermissionError('forbidden NPZ member '+key)
  # Never expose a NpzFile or an unrestricted member accessor to callers.
  with zipfile.ZipFile(self.path) as z:
   with z.open(key+'.npy','r') as f:array=np.load(io.BytesIO(f.read()),allow_pickle=False)
  self.member_reads.append(key);return array
 def dataset(self):
  x=self.array(KEYS[self.phase][0]);y=self.array(KEYS[self.phase][1])
  assert x.shape==(COUNTS[self.phase],64,64,3) and x.dtype==np.uint8
  assert y.shape in ((COUNTS[self.phase],1),(COUNTS[self.phase],)) and np.issubdtype(y.dtype,np.integer)
  y=y.reshape(-1).astype(np.int64);assert ((0<=y)&(y<8)).all()
  return x,y

def prepare_train(archive,destination):
 d=Path(destination);d.mkdir(parents=True,exist_ok=False);reader=MemberReader(archive);x,y=reader.dataset()
 # Population channel moments computed solely from official training pixels.
 total=np.zeros(3,dtype=np.float64);square=np.zeros(3,dtype=np.float64);count=0
 for start in range(0,len(x),64):
  a=x[start:start+64].astype(np.float64)/255.;total+=a.sum((0,1,2));square+=(a*a).sum((0,1,2));count+=a.shape[0]*64*64
 mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,0));assert (std>0).all()
 np.save(d/'train_images.npy',x,allow_pickle=False);np.save(d/'train_labels.npy',y,allow_pickle=False)
 receipt={'dataset':'BloodMNIST64','archive_path':str(Path(archive).resolve()),'archive_md5':digest(archive,'md5'),'archive_sha256':digest(archive),'train_samples':11959,'classes':8,'input_size':64,'layout':'NHWC RGB uint8','mean':mean.tolist(),'std':std.tolist(),'normalization_scope':'all official training pixels only; float64 population moments after /255','members_read':reader.member_reads,'validation_arrays_read':False,'test_arrays_read':False,'files':{n:digest(d/n) for n in ('train_images.npy','train_labels.npy')}}
 (d/'training_data_receipt.json').write_text(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--destination',required=True);a=p.parse_args();print(json.dumps(prepare_train(a.archive,a.destination)))
