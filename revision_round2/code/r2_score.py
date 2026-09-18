"""Locked full-set cross-scoring, with raw and symmetric BN-only states."""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import datasets
from r2_vision import ARCHIVE,WORK,VISION,configure,load_model,evaluate_transform,write_json,atomic_save,v
from r2_checks import calibrate_bn,calibration_inputs

LANG=ARCHIVE/'orthogonal_pilot_20260907'


def language(T,path=None):
    sys.path.insert(0,str(LANG/'code'))
    import paper_train as lang
    m=lang.model(T,path,train=False).eval()
    def forward(x):return m(input_ids=x,attention_mask=torch.ones_like(x)).logits
    return m,forward


def vision(T,path):
    m,sites,_,_=load_model(T,path,False)
    def forward(x):
        v.functional.reset_net(m);v.clear_costs(sites);return m(x).mean(0).float()
    return m,forward,sites


def training_inputs(task):
    if task=='language':
        d=torch.load(LANG/'data/paper_experiments_20260908/train_tune.pt',weights_only=True)
        return d['tune_x'],d['tune_y']
    d=json.loads((VISION/'data/CIFAR100_split.json').read_text())
    ds=datasets.CIFAR100(str(VISION/'data'),train=True,transform=evaluate_transform())
    ids=d['train'][:128]
    return torch.stack([ds[i][0] for i in ids]),torch.tensor([ds[i][1] for i in ids])


def final_inputs(task):
    assert (WORK/'results/model_analysis_lock.json').exists()
    if task=='language':
        d=torch.load(LANG/'data/paper_experiments_20260908/confirmation_inputs.pt',weights_only=True)['test']
        assert len(d['x'])==4406 and int((d['y']!=-100).sum())==40980
        return d['x'],d['y'],d['ids']
    ds=datasets.CIFAR100(str(VISION/'data'),train=False,transform=evaluate_transform())
    xs=torch.stack([ds[i][0] for i in range(len(ds))]);ys=torch.tensor(ds.targets)
    assert len(xs)==10000
    return xs,ys,torch.arange(len(xs))


@torch.no_grad()
def capture(forward,inputs):
    x=inputs[:4].cuda().clone()
    def eager():return forward(x)
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());tick=time.perf_counter()
    with torch.cuda.stream(side):
        for _ in range(5):z=eager()
    torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();del z
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):z=eager()
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    checks=[]
    for start in [0,64,0]:
        x.copy_(inputs[start:start+4]);expected=eager();graph.replay()
        checks.append({'offset':start,'bitwise_equal':torch.equal(expected,z),
           'max_logit_error':float((expected-z).abs().max()),'prediction_changes':int((expected.argmax(-1)!=z.argmax(-1)).sum())})
    # A nonzero numerical discrepancy is recorded, not hidden by skipping quality scoring.
    return x,graph,z,{'ABA':checks,'warmup_capture_seconds':seconds}


@torch.no_grad()
def score(row,data,calibration=None):
    configure();out=WORK/'results/scoring'/row['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    path=row['checkpoint'];T=row['infer_T'];task=row['task'];tick=time.perf_counter()
    if task=='language':m,forward=language(T,path)
    else:m,forward,sites=vision(T,path)
    meta={'queue_row':row,'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'precision':'fp32','tf32':False,'batch':4,'tail_rule':'repeat first valid row, score only original rows'}
    if row['calibration']=='bn_only':
        images,ids=calibration
        delta,record=calibrate_bn(m,sites,images)
        atomic_save({'bn_delta':delta,'source_checkpoint':path,'infer_T':T,'train_ids':ids},out/'bn_delta.pt')
        meta['bn_check']=record
    samples,_=training_inputs(task)
    xs,ys,ids=data
    if row['engine']=='graph':
        x,graph,z,checks=capture(forward,samples);meta['graph_check']=checks
        def call():graph.replay();return z
    else:
        x=samples[:4].cuda().clone()
        def call():return forward(x)
    sums=[];counts=[];correct=[];pred=[];labels=[]
    for start in range(0,len(xs),4):
        valid=min(4,len(xs)-start)
        x[:valid].copy_(xs[start:start+valid])
        if valid<4:x[valid:].copy_(xs[start:start+1].expand(4-valid,*xs.shape[1:]))
        logits=call()[:valid];y=ys[start:start+valid].cuda()
        if not torch.isfinite(logits).all():raise FloatingPointError(f'Nonfinite logits {row["id"]}/{start}')
        p=logits.argmax(-1)
        if task=='language':
            ce=nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
            mask=y!=-100
            sums.extend(ce.sum(-1).cpu().tolist());counts.extend(mask.sum(-1).cpu().tolist())
            correct.extend(((p==y)&mask).sum(-1).cpu().tolist())
            pred.extend(p[mask].cpu().tolist());labels.extend(y[mask].cpu().tolist())
        else:
            sums.extend(nn.functional.cross_entropy(logits,y,reduction='none').cpu().tolist())
            counts.extend([1]*valid);correct.extend((p==y).cpu().tolist())
            pred.extend(p.cpu().tolist());labels.extend(y.cpu().tolist())
    np.savez_compressed(out/'test.npz',block_id=np.asarray(ids),loss_sum=np.asarray(sums,dtype=np.float64),
        counts=np.asarray(counts,dtype=np.int64),correct=np.asarray(correct,dtype=np.int64),
        pred=np.asarray(pred,dtype=np.int64),labels=np.asarray(labels,dtype=np.int64))
    meta.update(status='complete',ce=float(np.sum(sums,dtype=np.float64)/sum(counts)),accuracy=float(sum(correct)/sum(counts)),
        items=len(xs),targets=sum(counts),process_seconds=time.perf_counter()-tick)
    write_json(out/'complete.json',meta)
    print(json.dumps({'id':row['id'],'ce':meta['ce'],'accuracy':meta['accuracy'],'seconds':meta['process_seconds']}),flush=True)
    return meta


@torch.no_grad()
def language_check():
    configure();xs,ys=training_inputs('language');rows=[]
    for T in [2,4]:
        m,forward=language(T)
        x,g,z,check=capture(forward,xs)
        natural=forward(xs[:2].cuda())
        padded=xs[:4].clone();padded[2:]=xs[:1]
        repeated=forward(padded.cuda())[:2]
        check.update(T=T,tail_shape_max_error=float((natural-repeated).abs().max()),
            tail_shape_prediction_changes=int((natural.argmax(-1)!=repeated.argmax(-1)).sum()))
        assert all(r['bitwise_equal'] for r in check['ABA']),check
        rows.append(check)
        del m,forward,g,z,x,natural,repeated;gc.collect();torch.cuda.empty_cache()
    write_json(WORK/'results/E0/language_migration.json',rows);print('LANGUAGE_CHECKS_PASSED',flush=True)


def worker(gpu):
    manifest=json.loads((WORK/'results/model_analysis_lock.json').read_text())
    data={};cal=None
    for row in manifest['scoring']:
        if row['gpu']!=gpu:continue
        if row['task'] not in data:data[row['task']]=final_inputs(row['task'])
        if row['calibration']=='bn_only' and cal is None:cal=calibration_inputs()
        try:score(row,data[row['task']],cal)
        except Exception:
            write_json(WORK/'results/scoring'/row['id']/'failure.json',{'status':'failed','error':traceback.format_exc(),'row':row})
            raise
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','worker']);ap.add_argument('--gpu',type=int)
    a=ap.parse_args()
    if a.action=='check':language_check()
    else:worker(a.gpu)
