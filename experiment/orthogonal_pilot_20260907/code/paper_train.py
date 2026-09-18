"""Frozen experiment training: train/tune only; no final-confirmation loader."""
from common import *
import argparse, platform

DEST=ROOT/'results/paper_experiments_20260908'
CACHE=ROOT/'data/paper_experiments_20260908'
DEST.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)

def training_data():
    p=CACHE/'train_tune.pt'
    if p.exists():return torch.load(p,weights_only=True)
    d=prepare_data();order=torch.randperm(len(d['train']),generator=torch.Generator().manual_seed(7600))
    x,y=mask_batch(d['train'][order[:128]],torch.Generator().manual_seed(7601))
    data={'train':d['train'][order[128:]],'tune_x':x,'tune_y':y,'tune_ids':order[:128]}
    torch.save(data,p);return data

def model(T,state=None,train=True):
    m=make_model('folded');m.bert.encoder.T=T
    if state is not None:
        cp=torch.load(state,map_location='cpu',weights_only=False)
        m.load_state_dict(cp.get('model',cp));del cp
    if not train:m.requires_grad_(False)
    return m

@torch.no_grad()
def assess(m,xs,ys,ids,path,batch=4):
    m.eval();ls=[];ns=[];cs=[];ps=[];labels=[]
    for i in range(0,len(xs),batch):
        x=xs[i:i+batch].cuda();y=ys[i:i+batch].cuda();z=m(input_ids=x,attention_mask=torch.ones_like(x)).logits
        ce=nn.functional.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
        mask=y!=-100;p=z.argmax(-1)
        ls.extend(ce.sum(-1).cpu().tolist());ns.extend(mask.sum(-1).cpu().tolist())
        cs.extend(((p==y)&mask).sum(-1).cpu().tolist());ps.extend(p[mask].cpu().tolist());labels.extend(y[mask].cpu().tolist())
    np.savez_compressed(path,block_id=np.asarray(ids),loss_sum=ls,counts=ns,correct=cs,pred=ps,labels=labels)
    return {'ce':sum(ls)/sum(ns),'accuracy':sum(cs)/sum(ns),'texts':len(xs),'masked_tokens':sum(ns)}

def rng_get(g):
    return {'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'data_generator':g.get_state()}

def rng_set(state,g):
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['cpu_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng']);g.set_state(state['data_generator'])

def atomic_save(obj,path):
    tmp=path.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(path)

def update(m,opt,params,d,g):
    torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True);value=0.
    for _ in range(4):
        ids=torch.randint(len(d['train']),(2,),generator=g);x,y=mask_batch(d['train'][ids],g)
        x=x.cuda();y=y.cuda();loss=m(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        (loss/4).backward();value+=float(loss.detach())/4
    norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
    return {'loss':value,'gradient_norm':float(norm),'seconds':time.perf_counter()-tick}

def train(args):
    tag=f'T{args.T}_seed{args.seed}_lr{args.lr:g}'
    out=DEST/args.stage/tag;out.mkdir(parents=True,exist_ok=True)
    cache=CACHE/args.stage/tag;cache.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('ALREADY_COMPLETE',tag,flush=True);return
    config={'T':args.T,'seed':args.seed,'lr':args.lr,'max_steps':args.steps,'stage':args.stage,
      'batch':8,'microbatch':2,'accumulation':4,'precision':'fp32','optimizer':'AdamW','weight_decay':0,'clip':1,
      'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
      'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cpu_threads':torch.get_num_threads()}
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    save_json(out/'config.json',config)
    m=model(args.T);d=training_data();params=[p for p in m.parameters() if p.requires_grad]
    save_json(out/'parameters.json',{'names':[n for n,p in m.named_parameters() if p.requires_grad],
        'count':sum(p.numel() for p in params),'frozen':[n for n,p in m.named_parameters() if not p.requires_grad]})
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0);seed_all(args.seed);g=torch.Generator().manual_seed(args.seed)
    step0=0;elapsed=0.;evaluations=[];latest=cache/'latest.pt'
    if latest.exists():
        cp=torch.load(latest,map_location='cpu',weights_only=False);m.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        step0=cp['step'];elapsed=cp['training_seconds'];evaluations=cp['evaluations'];rng_set(cp,g);del cp
        rows=[r for r in (out/'steps.jsonl').read_text().splitlines() if json.loads(r)['step']<=step0]
        (out/'steps.jsonl').write_text('\n'.join(rows)+'\n')
    m.train();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        with (out/'steps.jsonl').open('a',buffering=1) as log:
            for step in range(step0+1,args.steps+1):
                row=update(m,opt,params,d,g);elapsed+=row['seconds'];row.update(step=step,cumulative_training_seconds=elapsed)
                log.write(json.dumps(row)+'\n')
                if step%128==0:print(json.dumps({'run':tag,**row}),flush=True)
                if step in [512,1024,2048,4096] or step==args.steps:
                    state=rng_get(g);tick=time.perf_counter()
                    result=assess(m,d['tune_x'],d['tune_y'],d['tune_ids'],out/f'tune_step{step}.npz')
                    evaluations.append({'step':step,'training_seconds':elapsed,'tune':result,'evaluation_seconds':time.perf_counter()-tick})
                    save_json(out/'evaluations.json',evaluations);rng_set(state,g);m.train()
                    print(json.dumps({'run':tag,'step':step,'tune':result}),flush=True)
                if step%256==0 or step==args.steps:
                    tick=time.perf_counter();state=rng_get(g)
                    cp={'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'training_seconds':elapsed,
                        'evaluations':evaluations,'config':config,**state}
                    atomic_save(cp,latest)
                    if args.stage=='formal' or step==args.steps:
                        atomic_save({'model':m.state_dict(),'T':args.T,'seed':args.seed,'lr':args.lr,'step':step,
                            'training_seconds':elapsed,'stage':args.stage},cache/f'step{step}.pt')
                    save_json(out/'progress.json',{'step':step,'training_seconds':elapsed,'checkpoint_seconds':time.perf_counter()-tick,'status':'running'})
        save_json(out/'complete.json',{'config':config,'training_seconds':elapsed,'process_seconds':time.perf_counter()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'evaluations':evaluations,
            'checkpoint':str(cache/f'step{args.steps}.pt')})
        print('COMPLETE',tag,flush=True)
    except Exception as e:
        save_json(out/'failure.json',{'type':type(e).__name__,'message':str(e),'last_step':step,'completed_steps':step-1})
        raise

def check():
    out=DEST/'E1';out.mkdir(exist_ok=True);d=training_data();g=torch.Generator().manual_seed(8999)
    m=model(2);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0)
    seed_all(8999);m.train();update(m,opt,params,d,g)
    state=rng_get(g);tmp=CACHE/'resume_check.pt';atomic_save({'model':m.state_dict(),'optimizer':opt.state_dict(),**state},tmp)
    update(m,opt,params,d,g);expected={n:p.detach().cpu().clone() for n,p in m.state_dict().items()};gen_expected=g.get_state()
    optimizer_expected={k:{a:v.detach().cpu().clone() if torch.is_tensor(v) else v for a,v in val.items()} for k,val in opt.state_dict()['state'].items()}
    del m,opt,params;gc.collect();torch.cuda.empty_cache()
    m=model(2);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0)
    cp=torch.load(tmp,map_location='cpu',weights_only=False);m.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);rng_set(cp,g);del cp;m.train()
    update(m,opt,params,d,g)
    equal=all(torch.equal(p.cpu(),expected[n]) for n,p in m.state_dict().items())
    opt_equal=all(torch.equal(v.cpu(),optimizer_expected[k][a]) if torch.is_tensor(v) else v==optimizer_expected[k][a] for k,val in opt.state_dict()['state'].items() for a,v in val.items())
    result={'weight_bitwise_equal':equal,'optimizer_bitwise_equal':opt_equal,'sampling_generator_equal':torch.equal(g.get_state(),gen_expected)}
    save_json(out/'resume_check.json',result);print(json.dumps(result),flush=True)
    assert all(result.values()),result
    tmp.unlink()

def prepare():
    d=training_data();m=model(4,train=False);v=prepare_data();order=torch.randperm(len(v['validation']),generator=torch.Generator().manual_seed(6101))
    x,y=mask_batch(v['validation'][order[3424:3616]],torch.Generator().manual_seed(7602))
    out=DEST/'E0';out.mkdir(exist_ok=True);rows=[]
    for T in [2,4]:
        m.bert.encoder.T=T;r=assess(m,x,y,order[3424:3616],out/f'frozen_T{T}_old192.npz')
        old=np.load(ROOT/f'results/trained_state/baselines/T{T}.npz');new=np.load(out/f'frozen_T{T}_old192.npz')
        r.update(T=T,prediction_changes=int(np.sum(old['pred']!=new['pred'])),max_chunk_loss_error=float(np.max(np.abs(old['loss_sum']-new['loss_sum']))));rows.append(r)
    save_json(out/'initial_model_reproduction.json',rows)
    save_json(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'lif_backend':fused._LIF_BACKEND,'tf32':False,'train_chunks':len(d['train']),
        'tune_chunks':len(d['tune_x']),'validation_chunks':len(v['validation']),'token_length':64})
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','train']);p.add_argument('--T',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--lr',type=float);p.add_argument('--steps',type=int);p.add_argument('--stage',choices=['screen','formal','legacy','exploratory'])
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':check()
    else:train(a)
