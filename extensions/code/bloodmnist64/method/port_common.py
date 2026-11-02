import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[1]))
"""Fresh BloodMNIST64 authority; historical scientific primitives, new experiments."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCHEMA='bloodmnist64_covariant_tailq_half_20261030_v1_2'
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
def source_hash():
    from authority import verify_sources as check
    return check()
def verify_sources():return source_hash()
def software(root):
    from authority import is_software_fixture
    return is_software_fixture(root)
def verify_parent(bundle,root):
    from authority import B as PACKAGE
    bundle,root=Path(bundle).resolve(),Path(root).resolve();h=verify_sources()
    ctor_h=h
    if software(root):
        require(bundle==PACKAGE/'constructor','softwarefixture own isolated constructor')
    else:
        pin=read(PACKAGE/'CONSTRUCTOR_AUTHORITY.json');external=Path(pin['package']);require(bundle==external/'constructor','approved standalone constructor')
        ctor_h=sha(external/'SOURCE_FREEZE.json');require(ctor_h==pin['source_freeze_sha256'] and sha(bundle/'protocol.json')==pin['protocol_sha256'],'standalone constructor pins')
        for row in read(external/'SOURCE_FREEZE.json')['files']:require(sha(external/row['path'])==row['sha256'],'standalone constructor immutable source')
        for row in pin['method_runtime_dependencies']:require(sha(bundle/row['path'])==row['sha256'],'pinned original MTT runtime dependency')
    p=read(bundle/'protocol.json');require(p['data']['dataset']=='BloodMNIST64' and p['data']['classes']==8 and p['data']['input_size']==64,'explicit64/8dataset')
    prepared=read(root/'environment/blood64_prepared.json');require(prepared['protocol_sha256']==sha(bundle/'protocol.json') and prepared['source_freeze_sha256']==ctor_h,'prepared same frozen constructor source')
    data=Path(read(root/'environment/dataset.json')['data_path']);require(sha(data/'training_data_receipt.json')==prepared['data_receipt_sha256'],'current datareceipt bound to prepared source')
    for row in prepared['sources']:
        relative=Path(row['path']);require(sha(bundle/relative)==row['sha256'],'constructor sources unchanged')
        parts=relative.parts
        if len(parts)>=3 and parts[:2]==('payload','vision'):
            copied=root/'archive/direction1_budget_rotation_20260907'/Path(*parts[2:]);require(sha(copied)==row['sha256'],'copied vision payload unchanged')
        elif len(parts)>=3 and parts[:2]==('payload','legacy_code'):
            copied=root/'legacy/code'/Path(*parts[2:]);require(sha(copied)==row['sha256'],'copied legacy payload unchanged')
    from dataset import split
    require(read(root/'archive/direction1_budget_rotation_20260907/data/BloodMNIST64_split.json')==split(),'official full split globalIDs')
    role=read(root/'environment/construction_role.json');require(role['seed']==p['construction_roles'][role['role']] and role['role']==root.name,'role/seed/root identity')
    require(role['protocol_sha256']==sha(bundle/'protocol.json'),'construction protocol identity')
    return {'constructor_protocol_sha256':sha(bundle/'protocol.json'),'construction_root':str(root),'prepared_sha256':sha(root/'environment/blood64_prepared.json'),'construction_role_sha256':sha(root/'environment/construction_role.json'),'role':role['role'],'source_freeze_sha256':h,'schema':SCHEMA,'software_fixture':software(root)}

def verify_checkpoint(root,checkpoint,method):
    import torch
    root,checkpoint=Path(root).resolve(),Path(checkpoint).resolve();done=read(root/'construction/C-conf/complete.json')
    expected=root/'construction/C-conf/best.pt'
    role=read(root/'environment/construction_role.json')
    require(done['status']=='complete' and done['config']['dataset']=='BloodMNIST64' and (done['config']['epochs']==600 or (software(root) and done['config']['epochs']==0))
            and done['config']['construction_seed']==role['seed'] and done['checkpoint']==str(expected) and sha(expected)==done['checkpoint_sha256'],'complete own600epoch S0')
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if method=='S0':require(checkpoint==expected and sha(checkpoint)==done['checkpoint_sha256'],'retained checkpoint is actual own S0')
    elif method=='conditional':
        require(p['schema']==SCHEMA and p['native_S0_checkpoint_sha256']==done['checkpoint_sha256'],'conditional actual start identity')
        a=p['identity']['authority'];require(a['construction_root']==str(root) and a['prepared_sha256']==sha(root/'environment/blood64_prepared.json')
            and a['source_freeze_sha256']==source_hash(),'conditional actual source/root authority')
        terminal=read(checkpoint.parent/'complete.json')
        entries=terminal['checkpoints'];matches=[x for x in entries if x.get('mean',x.get('path'))==str(checkpoint)]
        require((terminal['status']=='complete' or (software(root) and terminal['status']=='acceptance_only' and terminal.get('software_fixture') is True)) and len(matches)==1 and matches[0].get('mean_sha256',matches[0].get('sha256'))==sha(checkpoint),'actual completed conditional checkpoint bytes')
    else:
        cfg=p['config'];require(cfg['initial_checkpoint']==str(expected) and cfg['initial_checkpoint_sha256']==done['checkpoint_sha256'],'native actual initial checkpoint')
        require(cfg['campaign_authority']['construction_root']==str(root) and cfg['campaign_authority']['source_freeze_sha256']==source_hash(),'native actual campaign authority')
        terminal=read(checkpoint.parents[2]/'training'/cfg['id']/'complete.json')
        matches=[x for x in terminal['checkpoints'] if x['checkpoint']==str(checkpoint)]
        require(terminal['status']=='complete' and len(matches)==1 and matches[0]['checkpoint_sha256']==sha(checkpoint),'actual complete baseline checkpoint bytes')
    return p

def verify_development_gate(path,paired_root):
    gate=read(path);require(gate['schema']==SCHEMA and gate['source_freeze_sha256']==source_hash(),'actual same method development gate')
    expected=Path(paired_root).resolve().parent/'development'
    require(gate['stage']=='development' and gate['construction_root']==str(expected)
            and gate['paired_seeds']==[34611,34612] and gate['units']==['dev34611','dev34612'],'actual paired development authority')
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
    require(frozen['schema']=='bloodmnist64_final_test_freeze_v1' and frozen['selection_complete'] is True,
            'final test selection must be frozen')
    require(frozen['method_source_freeze_sha256']==source_hash(),'actual final test method source')
    for name in ('rate_selection','development_gate','formal_gate'):
        require(sha(campaign/(name+'.json'))==frozen[name+'_sha256'],'actual frozen '+name)
    require(root.name in ('formal1','formal2','formal3'),'only independent formal starts enter test')
    verify_development_gate(campaign/'development_gate.json',root)
    from gate import bound_assess
    formal=read(campaign/'formal_gate.json')
    require(formal['gate_passed'] is True,'independent formal gate must pass before test')
    require(formal==bound_assess(formal['rows_path'],['formal34621','formal34622','formal34623'],
                               'formal',root.parent),'recomputed independent formal gate before test')
    require(len(frozen['checkpoints'])==39 and len({x['path'] for x in frozen['checkpoints']})==39,
            'all three formal starts and fixed selected models')
    require({'path':str(checkpoint),'sha256':sha(checkpoint),'method':method,
             'construction_root':str(root)} in frozen['checkpoints'],'exact frozen checkpoint and provenance')
    return frozen

def runtime(bundle,root):
    verify_sources();verify_parent(bundle,root)
    from authority import B as PACKAGE,require_production
    if not software(root):require_production(require_cross_gpu=True)
    bundle,root=Path(bundle).resolve(),Path(root).resolve();sys.path.insert(0,str(bundle/'scripts'))
    import common,runtime as rt
    require(Path(rt.__file__).resolve().parent==bundle/'scripts','isolatedBlood64runtime')
    common.environment(root);rt.configure();return RuntimeFacade(rt,root)
class RuntimeFacade:
    def __init__(self,rt,root):
        self.rt=rt;self.root=root;self._raw=None;self._planned=rt.planned_dataset;self._validation=rt.validation_loader;p=read(HERE.parent/'constructor/protocol.json');self.microbatch=p['execution']['microbatch'];self.calibration_batch=p['calibration']['batch'];self.evaluation_batch=p['execution']['evaluation_batch']
    def __getattr__(self,name):return getattr(self.rt,name)
    def raw(self):
        if self._raw is None:
            import dataset
            self._raw=dataset.read_arrays(self.rt.data_path(self.root),'train',self.root)
        return self._raw
    def image(self,i):
        from PIL import Image
        require(0<=int(i)<11959,'training globalID only')
        return Image.fromarray(self.raw()[0][int(i)])
    def planned_dataset(self,root,seed,updates,start=0):return self._planned(self.root,seed,updates,start)
    def calibrate(self,root,model,seed=9640):
        import numpy as np,torch
        from normalization import calibrate_bn
        ids=np.random.default_rng(seed).choice(np.asarray(self.rt.split(root)['train']),2048,replace=False);tf=self.evaluate_transform();images=torch.stack([tf(self.image(i)) for i in ids]);self.prepare_mode(model,False)
        bank,report=calibrate_bn(model,images,lambda:self.reset(model),lambda x:self.forward(model,x),batch=self.calibration_batch)
        report.update(train_ids=ids.tolist(),calibration_seed=seed,samples=2048,batches=2048//self.calibration_batch,protected_values_and_dtypes_unchanged=True);self.prepare_mode(model,False);return bank,report
    def validation_loader(self,root,batch=None):return self._validation(self.root,batch or self.evaluation_batch)
    def score_arrays(self,phase,test_context=None):
        import dataset,numpy as np
        x,y=dataset.read_arrays(self.rt.data_path(self.root),phase,self.root,test_context)
        return x,y,np.asarray(dataset.split()[phase],dtype=np.int64)

def guard_training(allow_validation=False):
    import os
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0]).replace('\\','/')
            import dataset
            permitted_val=allow_validation and dataset._VALIDATION_OPEN
            if ((path.endswith('/bloodmnist_64.npz') or path.endswith('/validation_images.npy') or path.endswith('/validation_labels.npy')) and not permitted_val) or path.endswith('/test_images.npy') or path.endswith('/test_labels.npy') or path.endswith('/scores.npz') or path.endswith('/replay_scores.npz'):
                raise RuntimeError('Training cannot access heldout scores/test labels: '+path)
    sys.addaudithook(audit)
