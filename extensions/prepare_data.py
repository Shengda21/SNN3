"""Reconstruct the fixed training inputs without reading heldout image tensors for normalization."""
import argparse,concurrent.futures,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def run(script,*args):subprocess.run([sys.executable,str(HERE/'data_tools'/script),*map(str,args)],check=True)
def materialize(root,dataset):
 import numpy as np
 from PIL import Image,ImageOps
 manifest=root/('proposed_split_v1.json' if dataset=='eurosat_rgb' else 'clean_split_proposal_v1_1.json');norm=root/('normalization_train_only_draft.json' if dataset=='eurosat_rgb' else 'normalization_clean_train_fullframe64_draft.json');d=read(manifest);n=read(norm);assert n['manifest_sha256']==sha(manifest)
 out=root/'execution_data_fullframe64_v1';out.mkdir(exist_ok=False);records=[];offset=0;counts={}
 for phase in ('train','validation','test'):
  rows=sorted((r for r in d['records'] if r['split']==phase),key=lambda r:r['global_id']);counts[phase]=len(rows)
  records.extend({**r,'source_global_id':r['global_id'],'global_id':offset+i,'phase_index':i} for i,r in enumerate(rows));offset+=len(rows)
 expected={'eurosat_rgb':{'train':18900,'validation':4050,'test':4050},'plantwild_v1':{'train':12741,'validation':1787,'test':3677}}[dataset];assert counts==expected
 train=[r for r in records if r['split']=='train'];x=np.lib.format.open_memmap(out/'train_images.npy',mode='w+',dtype=np.uint8,shape=(len(train),64,64,3));y=np.asarray([r['label'] for r in train],dtype=np.int64)
 def image(r):
  p=root/'original_extracted'/r['path'];assert sha(p)==r['file_sha256']
  with Image.open(p) as im:
   im=ImageOps.exif_transpose(im).convert('RGB')
   if dataset=='eurosat_rgb':assert im.size==(64,64)
   else:im=im.resize((64,64),Image.Resampling.BILINEAR)
   return np.asarray(im,dtype=np.uint8)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for i,arr in enumerate(pool.map(image,train)):x[i]=arr
 x.flush();del x;np.save(out/'train_labels.npy',y,allow_pickle=False)
 (out/'source_records.json').write_text(json.dumps({'classes':d['classes'],'records':records,'source_manifest_sha256':sha(manifest)},separators=(',',':')))
 receipt={'schema':'image64_training_data_v1','dataset':dataset,'software_fixture':False,'source_manifest_sha256':sha(manifest),'source_manifest_path':str(manifest),'source_root':str(root/'original_extracted'),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'mean':n['mean'],'std':n['std'],'counts':counts,'classes':len(d['classes']),'input_size':64,'normalization_sha256':sha(norm),'validation_images_read':False,'test_images_read':False}
 (out/'training_data_receipt.json').write_text(json.dumps(receipt,indent=2));print(out)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['bloodmnist64','eurosat','plantwild'],required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();archive=a.archive.resolve();root=a.destination.resolve()
 if a.dataset=='bloodmnist64':run('blood_train.py','--archive',archive,'--destination',root);return
 dataset='eurosat_rgb' if a.dataset=='eurosat' else 'plantwild_v1';assert root.name==dataset,'Use destination basename '+dataset
 run('prepare_cpu.py','--root',root,'--dataset',dataset,'--archive',archive);run('near_and_split.py','--root',root)
 if a.dataset=='plantwild':
  run('plant_official_split.py','--root',root);run('clean_plant_manifest.py','--root',root);manifest=root/'clean_split_proposal_v1_1.json';script='normalize_clean_fullframe64_cpu.py'
 else:manifest=root/'proposed_split_v1.json';script='normalize_train_cpu.py'
 run(script,'--manifest',manifest,'--root',root/'original_extracted');materialize(root,dataset)
if __name__=='__main__':main()
