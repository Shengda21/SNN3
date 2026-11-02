"""Fresh CIFAR-10 authority; historical scientific primitives, new experiments."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCHEMA='cifar100_covariant_tailq_half_20261030_v1_2'
VIEWS=('2_raw','2_matched','4_raw','4_matched')
BUDGETS=(512,1024,2048)
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify_sources():
    frozen=read(HERE/'SOURCE_FREEZE.json')
    require(frozen['schema']==SCHEMA,'own frozen source schema')
    for name,digest in frozen['files'].items():require(sha(HERE/name)==digest,'changed frozen source '+name)

def verify_parent(bundle,root):
    bundle,root=Path(bundle).resolve(),Path(root).resolve();authority=read(HERE/'PARENT_CONSTRUCTOR.json')
    manifest=Path(authority['manifest_path']);require(sha(manifest)==authority['manifest_sha256'],'actual staged constructor manifest')
    records=read(manifest);require(records==authority['files'] and len(records)==authority['source_count'],'exact complete actual RTX5090 constructor sources')
    for row in records:require(sha(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes'],'actual frozen constructor source '+row['path'])
    own={Path(x['path']).relative_to(bundle).as_posix():x['sha256'] for x in records if Path(x['path']).is_relative_to(bundle)}
    require(own and sha(bundle/'protocol.json')==authority['protocol_sha256'],'actual constructor bundle/protocol')
    prepared=read(root/'environment/cifar100_prepared.json')
    require(prepared['protocol_sha256']==authority['protocol_sha256'] and {x['path']:x['sha256'] for x in prepared['sources']}==own,'actual own prepared authority')
    vision=root/'archive/direction1_budget_rotation_20260907'
    for name,digest in own.items():
        if name.startswith('payload/vision/'):target=vision/name.removeprefix('payload/vision/')
        elif name.startswith('payload/legacy_code/'):target=root/'legacy/code'/name.removeprefix('payload/legacy_code/')
        else:continue
        require(sha(target)==digest,'prepared copied runtime source '+str(target))
    split_path=vision/'data/CIFAR100_split.json';require(sha(split_path)==prepared['split_sha256'],'prepared frozen split bytes')
    import numpy as np
    split=read(split_path);ids=np.random.default_rng(20261030).permutation(50000)
    require(split['dataset']=='CIFAR100' and split['train']==ids[:45000].tolist() and split['validation']==ids[45000:].tolist(),'original complete frozen CIFAR100 split')
    role=read(root/'environment/construction_role.json');protocol=read(bundle/'protocol.json')
    require(role['protocol_sha256']==authority['protocol_sha256'] and role['seed']==protocol['construction_roles'][role['role']] and role.get('execution_scope')=='reader_replication','new constructor role authority')
    require(role['role']==root.name and root.name in ('development','formal1','formal2','formal3'),'actual fixed construction-root role')
    return {'constructor_manifest_sha256':authority['manifest_sha256'],'constructor_protocol_sha256':authority['protocol_sha256'],
            'construction_root':str(root),'prepared_sha256':sha(root/'environment/cifar100_prepared.json'),
            'construction_role_sha256':sha(root/'environment/construction_role.json'),'role':role['role'],
            'source_freeze_sha256':sha(HERE/'SOURCE_FREEZE.json'),'schema':SCHEMA}

def verify_checkpoint(root,checkpoint,method):
    import torch
    root,checkpoint=Path(root).resolve(),Path(checkpoint).resolve();done=read(root/'construction/C-conf/complete.json')
    expected=root/'construction/C-conf/best.pt'
    role=read(root/'environment/construction_role.json')
    require(done['status']=='complete' and done['config']['dataset']=='CIFAR100' and done['config']['epochs']==600
            and done['config']['construction_seed']==role['seed'] and done['checkpoint']==str(expected) and sha(expected)==done['checkpoint_sha256'],'complete own600epoch S0')
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if method=='S0':require(checkpoint==expected and sha(checkpoint)==done['checkpoint_sha256'],'retained checkpoint is actual own S0')
    elif method=='conditional':
        require(p['schema']==SCHEMA and p['native_S0_checkpoint_sha256']==done['checkpoint_sha256'],'conditional actual start identity')
        a=p['identity']['authority'];require(a['construction_root']==str(root) and a['prepared_sha256']==sha(root/'environment/cifar100_prepared.json')
            and a['source_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json'),'conditional actual source/root authority')
        terminal=read(checkpoint.parent/'complete.json')
        entries=terminal['checkpoints'];matches=[x for x in entries if x.get('mean',x.get('path'))==str(checkpoint)]
        require(terminal['status']=='complete' and len(matches)==1 and matches[0].get('mean_sha256',matches[0].get('sha256'))==sha(checkpoint),'actual completed conditional checkpoint bytes')
    else:
        cfg=p['config'];require(cfg['initial_checkpoint']==str(expected) and cfg['initial_checkpoint_sha256']==done['checkpoint_sha256'],'native actual initial checkpoint')
        require(cfg['campaign_authority']['construction_root']==str(root) and cfg['campaign_authority']['source_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json'),'native actual campaign authority')
        terminal=read(checkpoint.parents[2]/'training'/cfg['id']/'complete.json')
        matches=[x for x in terminal['checkpoints'] if x['checkpoint']==str(checkpoint)]
        require(terminal['status']=='complete' and len(matches)==1 and matches[0]['checkpoint_sha256']==sha(checkpoint),'actual complete baseline checkpoint bytes')
    return p

def verify_development_gate(path,paired_root):
    gate=read(path);require(gate['schema']==SCHEMA and gate['source_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json'),'actual same method development gate')
    expected=Path(paired_root).resolve().parent/'development'
    require(gate['stage']=='development' and gate['construction_root']==str(expected)
            and gate['paired_seeds']==[23611,23612] and gate['units']==['dev23611','dev23612'],'actual paired development authority')
    for row in gate['score_sources']:require(sha(row['path'])==row['sha256'],'actual development score unchanged')
    require(sha(gate['rows_path'])==gate['rows_sha256'],'actual paired development rows')
    from gate import bound_assess
    recomputed=bound_assess(gate['rows_path'],gate['units'],'development',expected)
    require(gate==recomputed,'independently recomputed complete development gate')
    for field in ('endpoint_gate_passed','full_budget_accuracy_target_met','full_budget_CE_target_met','gate_passed'):
        require(gate[field] is True,'original complete development gate '+field)
    require(len(gate['conditions'])==24 and len(gate['score_sources'])==56,'complete development matrix')
    return gate

def verify_test_freeze(path,checkpoint,root,method):
    path,checkpoint,root=Path(path).resolve(),Path(checkpoint).resolve(),Path(root).resolve()
    frozen=read(path);campaign=path.parent
    require(frozen['schema']=='cifar100_final_test_freeze_v1' and frozen['selection_complete'] is True,
            'final test selection must be frozen')
    require(frozen['method_source_freeze_sha256']==sha(HERE/'SOURCE_FREEZE.json'),'actual final test method source')
    for name in ('rate_selection','development_gate','formal_gate'):
        require(sha(campaign/(name+'.json'))==frozen[name+'_sha256'],'actual frozen '+name)
    require(root.name in ('formal1','formal2','formal3'),'only independent formal starts enter test')
    verify_development_gate(campaign/'development_gate.json',root)
    from gate import bound_assess
    formal=read(campaign/'formal_gate.json')
    require(formal['gate_passed'] is True,'independent formal gate must pass before test')
    require(formal==bound_assess(formal['rows_path'],['formal23651','formal23652','formal23653'],
                               'formal',root.parent),'recomputed independent formal gate before test')
    require(len(frozen['checkpoints'])==39 and len({x['path'] for x in frozen['checkpoints']})==39,
            'all three formal starts and fixed selected models')
    require({'path':str(checkpoint),'sha256':sha(checkpoint),'method':method,
             'construction_root':str(root)} in frozen['checkpoints'],'exact frozen checkpoint and provenance')
    return frozen

def runtime(bundle,root):
    verify_sources();verify_parent(bundle,root)
    bundle,root=Path(bundle).resolve(),Path(root).resolve()
    sys.path.insert(0,str(bundle/'scripts'))
    import common,runtime as rt
    require(Path(rt.__file__).resolve().parent==bundle/'scripts','isolated CIFAR100 runtime')
    require(read(bundle/'protocol.json')['data']['dataset']=='CIFAR100','CIFAR100-only campaign')
    common.environment(root)
    require(read(root/'environment/construction_role.json').get('execution_scope')=='reader_replication','new authorized RTX5090 construction')
    rt.configure();return RuntimeFacade(rt,root)
class RuntimeFacade:
    def __init__(self,rt,root):self.rt=rt;self.root=root;self._raw=None
    def __getattr__(self,name):return getattr(self.rt,name)
    def raw(self):
        if self._raw is None:
            import pickle,numpy as np
            path=self.rt.vision_root(self.root)/'data/cifar-100-python/train'
            from torchvision.datasets import CIFAR100
            with path.open('rb') as f:require(hashlib.file_digest(f,'md5').hexdigest()==dict(CIFAR100.train_list)['train'],'official CIFAR100 training batch')
            with path.open('rb') as f:value=pickle.load(f,encoding='latin1')
            self._raw=np.asarray(value['data']),np.asarray(value['fine_labels'],dtype=np.int64)
        return self._raw
    def image(self,i):
        from PIL import Image
        return Image.fromarray(self.raw()[0][int(i)].reshape(3,32,32).transpose(1,2,0))
    def planned_dataset(self,root,seed,updates,start=0):
        facade=self
        import random,numpy as np
        import torch
        class Planned:
            def __init__(self):
                self.indices,self.aug_seeds=facade.rt.sample_plan(root,seed,updates);self.start=start*128
                self.transform=facade.rt.training_transform();facade.raw()
            def __len__(self):return len(self.indices)-self.start
            def __getitem__(self,n):
                n+=self.start;s=int(self.aug_seeds[n]);random.seed(s);np.random.seed(s);torch.random.default_generator.manual_seed(s)
                i=int(self.indices[n]);return self.transform(facade.image(i)),int(facade.raw()[1][i])
        return Planned()
    def calibrate(self,root,model,seed=9640):
        import numpy as np,torch
        from normalization import calibrate_bn
        ids=np.random.default_rng(seed).choice(np.asarray(self.rt.split(root)['train']),2048,replace=False)
        transform=self.rt.evaluate_transform();images=torch.stack([transform(self.image(i)) for i in ids])
        self.prepare_mode(model,False)
        bank,report=calibrate_bn(model,images,lambda:self.reset(model),lambda x:self.forward(model,x),batch=64)
        report.update(train_ids=ids.tolist(),calibration_seed=seed,samples=2048,batches=32,
                      protected_values_and_dtypes_unchanged=True)
        self.prepare_mode(model,False);return bank,report
    def validation_loader(self,root,batch=64):
        import torch
        facade=self;indices=self.rt.split(self.root)['validation'];transform=self.rt.evaluate_transform();self.raw()
        class Validation:
            def __len__(self):return len(indices)
            def __getitem__(self,n):
                i=int(indices[n]);return transform(facade.image(i)),int(facade.raw()[1][i])
        return torch.utils.data.DataLoader(Validation(),batch_size=batch,shuffle=False,num_workers=4,pin_memory=True)
def guard_training():
    import os
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0]).replace('\\','/')
            if (path.endswith('/test_batch') or '/cifar-100-python/test' in path) or path.endswith('/scores.npz') or path.endswith('/replay_scores.npz'):
                raise RuntimeError('Training cannot access heldout scores/test labels: '+path)
    sys.addaudithook(audit)
