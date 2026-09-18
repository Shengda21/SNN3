"""Paired eager/graph execution, fixed-shape A/B/A, separate D2D-copy scope."""
from paper_train import *
import argparse

def language(T,checkpoint):
    m=model(T,checkpoint,train=False).eval();d=training_data();samples=d['tune_x'].cuda()
    return m,samples,d,lambda x:m(input_ids=x,attention_mask=torch.ones_like(x)).logits

def vision(T,checkpoint):
    old=ROOT.parent/'direction1_budget_rotation_20260907';sys.path.insert(0,str(old/'code'))
    import train as v
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=cp['config']
    m,sites=v.build_model(cfg);v.load_preserving_buffers(m,cp['model']);m.T=T;m.eval().requires_grad_(False);v.prepare_inference(m)
    from torchvision import datasets,transforms
    ds=datasets.CIFAR100(str(old/'data'),train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))]))
    idx=json.loads((old/'data/CIFAR100_split.json').read_text())['train'][:128]
    samples=torch.stack([ds[i][0] for i in idx]).cuda()
    def call(x):v.functional.reset_net(m);v.clear_costs(sites);return m(x).mean(0).float()
    val_ids=json.loads((old/'data/CIFAR100_split.json').read_text())['validation'][:1024]
    old_data={'old_x':torch.stack([ds[i][0] for i in val_ids]),'old_y':torch.tensor([ds[i][1] for i in val_ids])}
    return m,samples,old_data,call

@torch.no_grad()
def run(a):
    out=DEST/'benchmark'/a.group/f'{a.architecture}_{a.tag}_session{a.session}';out.mkdir(parents=True,exist_ok=True)
    order=[2,4] if a.session%2 else [4,2]
    if a.T:order=[a.T]
    for T in order:
        m,samples,d,forward=(language(T,a.checkpoint) if a.architecture=='language' else vision(T,a.checkpoint))
        for b in a.batches:
            for route in (['full','masked'] if a.masked else ['full']):
                name=f'T{T}_b{b}_{route}';path=out/f'{name}.json'
                if path.exists() and json.loads(path.read_text()).get('status')=='complete':continue
                meta={'architecture':a.architecture,'checkpoint':a.checkpoint or 'original_frozen','tag':a.tag,'T':T,'batch':b,
                      'route':route,'session':a.session,'gpu':torch.cuda.get_device_name(),'gpu_id':os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'torch':torch.__version__,'cuda':torch.version.cuda,'fp32':True,'tf32':False,'copy_scope':'D2D input ids/image; attention mask constant'}
                try:
                    x=samples[:b].clone();source=samples[:b].clone()
                    if a.architecture=='language':
                        att=torch.ones_like(x)
                        positions=(d['tune_y'][:b].flatten()!=-100).nonzero().flatten().cuda()
                        if route=='masked':
                            def eager():
                                h=m.bert(input_ids=x,attention_mask=att).last_hidden_state.flatten(0,1).index_select(0,positions)
                                return m.cls(h)
                        elif a.masked:
                            def eager():return m(input_ids=x,attention_mask=att).logits.flatten(0,1).index_select(0,positions)
                        else:
                            def eager():return m(input_ids=x,attention_mask=att).logits
                        meta['selected_tokens']=int(positions.numel())
                    else:
                        def eager():return forward(x)
                    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());tick=time.perf_counter()
                    with torch.cuda.stream(side):
                        for _ in range(5):z=eager()
                    torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();del z
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):static_z=eager()
                    torch.cuda.synchronize();meta['warmup_capture_seconds']=time.perf_counter()-tick
                    checks=[]
                    for offset in [0,64,0]:
                        x.copy_(samples[offset:offset+b]);expected=eager();graph.replay()
                        checks.append({'offset':offset,'bitwise_equal':bool(torch.equal(expected,static_z)),
                            'max_abs_logit_error':float((expected-static_z).abs().max()),'prediction_changes':int((expected.argmax(-1)!=static_z.argmax(-1)).sum())})
                        del expected
                    meta['checks']=checks
                    if not all(c['bitwise_equal'] for c in checks):
                        meta['status']='output_preservation_failed';save_json(path,meta);del graph,static_z;continue
                    if b==4 and not a.masked:
                        if a.architecture=='language':
                            raw=prepare_data();ix=torch.randperm(len(raw['validation']),generator=torch.Generator().manual_seed(6101))[3424:3616]
                            cx,cy=mask_batch(raw['validation'][ix],torch.Generator().manual_seed(7602))
                        else:cx,cy=d['old_x'],d['old_y']
                        maxerr=0.;flips=0;equal=True;loss_sum=0.;correct=0;count=0
                        for start in range(0,len(cx),b):
                            x.copy_(cx[start:start+b].cuda());labels=cy[start:start+b].cuda();ez=eager();graph.replay()
                            equal=equal and bool(torch.equal(ez,static_z));maxerr=max(maxerr,float((ez-static_z).abs().max()))
                            flips+=int((ez.argmax(-1)!=static_z.argmax(-1)).sum())
                            if a.architecture=='language':
                                valid=labels!=-100;loss_sum+=float(nn.functional.cross_entropy(static_z.flatten(0,1),labels.flatten(),reduction='sum'))
                                correct+=int(((static_z.argmax(-1)==labels)&valid).sum());count+=int(valid.sum())
                            else:
                                loss_sum+=float(nn.functional.cross_entropy(static_z,labels,reduction='sum'));correct+=int((static_z.argmax(-1)==labels).sum());count+=len(labels)
                            del ez
                        meta['old_confirmation']={'items':len(cx),'bitwise_equal':equal,'max_logit_error':maxerr,'prediction_changes':flips,'ce':loss_sum/count,'accuracy':correct/count}
                        if not equal:
                            meta['status']='full_confirmation_output_preservation_failed';save_json(path,meta);del graph,static_z;continue
                    x.copy_(source)
                    def replay():graph.replay();return static_z
                    def copy_replay():x.copy_(source);graph.replay();return static_z
                    def copy_eager():x.copy_(source);return eager()
                    calls={'eager':eager,'graph':replay}
                    if not a.masked:calls['graph_with_copy']=copy_replay
                    if a.copy_control and b in [1,4]:calls['eager_with_copy']=copy_eager
                    rows=[];keys=list(calls)
                    for rnd in range(3):
                        k=(a.session+rnd)%len(keys);sequence=keys[k:]+keys[:k]
                        if rnd%2:sequence=sequence[::-1]
                        for kind in sequence:
                            fn=calls[kind]
                            for _ in range(5):z=fn();del z
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                            for rep in range(10):
                                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                torch.cuda.synchronize();tick=time.perf_counter();begin.record();z=fn();end.record();end.synchronize()
                                wall=1000*(time.perf_counter()-tick)
                                rows.append({'engine':kind,'round':rnd,'repeat':rep,'wall_ms':wall,'cuda_ms':begin.elapsed_time(end),
                                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.memory_reserved()/2**30});del z
                    meta.update(status='complete',measurements=rows,medians={k:float(np.median([r['wall_ms'] for r in rows if r['engine']==k])) for k in keys})
                    save_json(path,meta);print(json.dumps({'configuration':name,'tag':a.tag,'medians':meta['medians']}),flush=True)
                    del calls,graph,static_z,x,source;gc.collect();torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError as e:
                    meta.update(status='oom',error=str(e));save_json(path,meta);raise
        del m,forward,samples;gc.collect();torch.cuda.empty_cache()
    save_json(out/'complete.json',{'completed':True,'session':a.session,'tag':a.tag})

@torch.no_grad()
def masked_quality():
    d=prepare_data();idx=torch.randperm(len(d['validation']),generator=torch.Generator().manual_seed(6101))[3424:3616]
    xs,ys=mask_batch(d['validation'][idx],torch.Generator().manual_seed(7602));out=DEST/'E4c';out.mkdir(exist_ok=True)
    m=model(2,train=False).eval();rows=[]
    for T in [2,4]:
        m.bert.encoder.T=T;ls=[];ns=[];cs=[];pred=[];maxerr=0.;flips=0
        for i in range(0,192,4):
            x=xs[i:i+4].cuda();y=ys[i:i+4].cuda();mask=y!=-100
            h=m.bert(input_ids=x,attention_mask=torch.ones_like(x)).last_hidden_state
            full=m.cls(h)[mask];selected=m.cls(h[mask]);maxerr=max(maxerr,float((full-selected).abs().max()));flips+=int((full.argmax(-1)!=selected.argmax(-1)).sum())
            ce=nn.functional.cross_entropy(selected,y[mask],reduction='none');owner=torch.arange(4,device='cuda')[:,None].expand_as(y)[mask]
            for j in range(4):
                valid=owner==j;ls.append(float(ce[valid].sum()));ns.append(int(valid.sum()));cs.append(int((selected[valid].argmax(-1)==y[mask][valid]).sum()))
            pred.extend(selected.argmax(-1).cpu().tolist())
        np.savez_compressed(out/f'T{T}_old192.npz',block_id=idx,loss_sum=ls,counts=ns,correct=cs,pred=pred)
        rows.append({'T':T,'ce':sum(ls)/sum(ns),'accuracy':sum(cs)/sum(ns),'max_logit_error':maxerr,'prediction_changes':flips,'task_preservation_gate':maxerr<1e-3 and flips==0})
    save_json(out/'quality.json',rows);print(json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['language','vision'],default='language');p.add_argument('--checkpoint');p.add_argument('--T',type=int)
    p.add_argument('--session',type=int,default=8801);p.add_argument('--batches',type=int,nargs='+',default=[1,4,16,64]);p.add_argument('--group',default='historical');p.add_argument('--tag',default='frozen')
    p.add_argument('--masked',action='store_true');p.add_argument('--copy-control',action='store_true');p.add_argument('--masked-quality',action='store_true');a=p.parse_args()
    if a.masked_quality:masked_quality()
    else:run(a)
