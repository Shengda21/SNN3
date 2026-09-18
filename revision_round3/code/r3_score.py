"""Fixed-batch scoring, including saved BN statistics reused across windows."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import datasets
from r2_vision import WORK,VISION,configure,load_model,evaluate_transform,write_json,atomic_save,state_cpu,v
from r2_score import language,training_inputs,capture
from r2_checks import calibrate_bn

LEGACY=Path(os.environ.get('R3_LEGACY_WORK','/root/snn_revision_round2'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def inputs(task):
    if task=='language':
        from r2_score import LANG
        d=torch.load(LANG/'data/paper_experiments_20260908/confirmation_inputs.pt',weights_only=True)['test']
        return d['x'],d['y'],d['ids']
    ds=datasets.CIFAR100(str(VISION/'data'),train=False,transform=evaluate_transform())
    return torch.stack([ds[i][0] for i in range(len(ds))]),torch.tensor(ds.targets),torch.arange(len(ds))
def calibration(seed):
    split=json.loads((VISION/'data/CIFAR100_split.json').read_text())
    ids=np.random.default_rng(seed).choice(np.asarray(split['train']),2048,replace=False)
    ds=datasets.CIFAR100(str(VISION/'data'),train=True,transform=evaluate_transform())
    return torch.stack([ds[int(i)][0] for i in ids]),ids

@torch.no_grad()
def score(row,data,cal_data,check_data):
    out=WORK/'results/scoring'/row['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    start_time=time.perf_counter();task=row['task'];e=row['infer_T']
    if task=='language':m,forward=language(e,row.get('checkpoint'));sites=None
    else:
        c=row.get('cal_T',e)
        m,sites,_,_=load_model(c,row['checkpoint'],False)
        def forward(x):
            v.functional.reset_net(m);v.clear_costs(sites);return m(x).mean(0).float()
    meta={'row':row,'status':'running','batch':4,'precision':'fp32','tf32':False,
          'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
          'tail_rule':'repeat first valid row, count only original rows'}
    if task=='vision' and row['calibration']!='raw':
        images,ids=cal_data[row['cal_seed']]
        identity=row['checkpoint_sha256']
        dest=WORK/'models/calibration'/f"{identity[:16]}_c{row['cal_T']}_subset{row['cal_seed']}.pt"
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            saved=torch.load(dest,map_location='cpu',weights_only=False)
            assert saved['checkpoint_sha256']==identity and saved['cal_T']==row['cal_T'] and np.array_equal(saved['train_ids'],ids)
            before=state_cpu(m);current=m.state_dict()
            for name,tensor in saved['bn_delta'].items():
                assert name.endswith(('.running_mean','.running_var','.num_batches_tracked'))
                assert current[name].dtype==tensor.dtype
                current[name].copy_(tensor)
            after=state_cpu(m)
            assert all(torch.equal(before[k],after[k]) and before[k].dtype==after[k].dtype for k in before if k not in saved['bn_delta'])
            record={'loaded_saved_statistics':True,'protected_values_and_dtypes_unchanged':True,'samples':2048,'bn_layers':33}
            del before,after,current,saved
        else:
            delta,record=calibrate_bn(m,sites,images)
            atomic_save({'bn_delta':delta,'checkpoint_sha256':identity,'cal_T':row['cal_T'],
                         'cal_seed':row['cal_seed'],'train_ids':ids},dest)
        meta['bn_check']=record;meta['bn_state_file']=str(dest)
        # No recalibration at e: this is the actual off-diagonal state-reuse operation.
        m.T=e;v.functional.reset_net(m);v.clear_costs(sites);v.prepare_inference(m)
    requested=row.get('engine','graph')
    if requested=='graph':
        x,graph,z,record=capture(forward,check_data[task])
        meta['graph_check']=record
        if all(q['bitwise_equal'] for q in record['ABA']):
            def call():graph.replay();return z
            actual_engine='graph'
        else:
            del graph,z
            def call():return forward(x)
            actual_engine='eager'
            meta['graph_fallback_reason']='Training-side ABA outputs were not bitwise equal.'
    else:
        x=check_data[task][:4].cuda().clone()
        def call():return forward(x)
        actual_engine='eager'
    xs,ys,ids=data
    xs=xs.cuda();ys=ys.cuda()
    sums=[];counts=[];correct=[];pred=[];labels=[]
    for start in range(0,len(xs),4):
        n=min(4,len(xs)-start);x[:n].copy_(xs[start:start+n])
        if n<4:x[n:].copy_(xs[start:start+1].expand(4-n,*xs.shape[1:]))
        logits=call()[:n];y=ys[start:start+n];predicted=logits.argmax(-1)
        if task=='language':
            ce=nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
            mask=y!=-100
            sums.append(ce.sum(-1).clone());counts.append(mask.sum(-1).clone())
            correct.append(((predicted==y)&mask).sum(-1).clone())
            pred.append(predicted[mask].clone());labels.append(y[mask].clone())
        else:
            sums.append(nn.functional.cross_entropy(logits,y,reduction='none').clone())
            counts.append(torch.ones(n,device='cuda',dtype=torch.long))
            correct.append((predicted==y).long().clone());pred.append(predicted.clone());labels.append(y.clone())
    arrays={'block_id':np.asarray(ids)}
    for name,parts in [('loss_sum',sums),('counts',counts),('correct',correct),('pred',pred),('labels',labels)]:
        arrays[name]=torch.cat(parts).cpu().numpy().astype(np.float64 if name=='loss_sum' else np.int64)
    assert np.isfinite(arrays['loss_sum']).all()
    np.savez_compressed(out/'test.npz',**arrays)
    meta.update(status='complete',engine=actual_engine,ce=float(arrays['loss_sum'].sum()/arrays['counts'].sum()),
                accuracy=float(arrays['correct'].sum()/arrays['counts'].sum()),items=len(xs),targets=int(arrays['counts'].sum()),
                process_seconds=time.perf_counter()-start_time)
    if 'legacy_id' in row:
        old=np.load(LEGACY/'results/scoring'/row['legacy_id']/'test.npz')
        meta['legacy_comparison']={'mean_ce_change':meta['ce']-float(old['loss_sum'].sum()/old['counts'].sum()),
             'max_item_loss_change':float(np.max(np.abs(arrays['loss_sum']-old['loss_sum']))),
             'prediction_changes':int(np.sum(arrays['pred']!=old['pred'])),
             'targets_and_labels_equal':bool(np.array_equal(arrays['labels'],old['labels']) and np.array_equal(arrays['counts'],old['counts']))}
        assert meta['legacy_comparison']['targets_and_labels_equal']
    write_json(out/'complete.json',meta)
    print(json.dumps({'id':row['id'],'ce':meta['ce'],'accuracy':meta['accuracy'],'engine':actual_engine,'seconds':meta['process_seconds']}),flush=True)
    return meta

def worker(rows):
    configure();data={};cal_data={};check_data={}
    for i,row in enumerate(rows):
        if (WORK/'results/scoring'/row['id']/'complete.json').exists():continue
        task=row['task']
        if task not in data:data[task]=inputs(task);check_data[task]=training_inputs(task)[0]
        if row['calibration']!='raw' and row['cal_seed'] not in cal_data:cal_data[row['cal_seed']]=calibration(row['cal_seed'])
        try:score(row,data[task],cal_data,check_data)
        except Exception:
            import traceback
            write_json(WORK/'results/scoring'/row['id']/'failure.json',{'row':row,'error':traceback.format_exc()})
            raise
        gc.collect();torch.cuda.empty_cache()
        write_json(WORK/'results/scoring_status.json',{'done':i+1,'total':len(rows),'last':row['id'],'updated':time.time()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    a=p.parse_args();rows=json.loads(Path(a.queue).read_text())
    sources=sorted({r['checkpoint_sha256'] for r in rows})
    assigned=set(sources[a.part::a.parts])
    worker([r for r in rows if r['checkpoint_sha256'] in assigned])
