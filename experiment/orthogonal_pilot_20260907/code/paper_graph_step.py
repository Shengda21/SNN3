"""Optional exact-checked training graph. CPU sampling and AdamW remain unchanged."""
import gc,time,json
import torch
from paper_train import model,training_data,seed_all,rng_get,rng_set,update,nn,mask_batch,DEST,CACHE,atomic_save,save_json

class GraphStep:
    def __init__(self,m,opt,params,d,g):
        self.m=m;self.opt=opt;self.params=params;self.d=d;self.g=g
        state=rng_get(g);self.x=torch.zeros((8,64),dtype=torch.long,device='cuda');self.y=torch.zeros_like(self.x)
        self.fill()
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):self.forward_backward()
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.losses=self.forward_backward()
        torch.cuda.synchronize();rng_set(state,g)

    def fill(self):
        xs=[];ys=[]
        for _ in range(4):
            idx=torch.randint(len(self.d['train']),(2,),generator=self.g);x,y=mask_batch(self.d['train'][idx],self.g);xs.append(x);ys.append(y)
        self.x.copy_(torch.cat(xs));self.y.copy_(torch.cat(ys))

    def forward_backward(self):
        self.opt.zero_grad(set_to_none=False);losses=[]
        for i in range(4):
            x=self.x[i*2:(i+1)*2];y=self.y[i*2:(i+1)*2]
            loss=self.m(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss
            (loss/4).backward();losses.append(loss.detach())
        return torch.stack(losses)

    def __call__(self,m,opt,params,d,g):
        torch.cuda.synchronize();start=time.perf_counter();self.fill();self.graph.replay()
        if not torch.isfinite(self.losses).all():raise FloatingPointError('Nonfinite graph loss')
        norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        return {'loss':float(self.losses.double().mean()),'gradient_norm':float(norm),'seconds':time.perf_counter()-start}

def check():
    d=training_data();rows=[]
    for T in [2,4]:
        m=model(T);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0)
        seed_all(8988);g=torch.Generator().manual_seed(8988);initial=rng_get(g)
        init=CACHE/f'graph_check_initial_T{T}.pt';atomic_save({'model':m.state_dict()},init);m.train()
        eager_times=[]
        for _ in range(3):eager_times.append(update(m,opt,params,d,g)['seconds'])
        expected={n:p.detach().cpu().clone() for n,p in m.state_dict().items()};gen=g.get_state()
        optim_expected={k:{a:v.detach().cpu().clone() if torch.is_tensor(v) else v for a,v in vs.items()} for k,vs in opt.state_dict()['state'].items()}
        del m,params,opt;gc.collect();torch.cuda.empty_cache()
        m=model(T,init);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0);m.train();rng_set(initial,g)
        fn=GraphStep(m,opt,params,d,g);graph_times=[]
        for _ in range(3):graph_times.append(fn(m,opt,params,d,g)['seconds'])
        unequal=[n for n,p in m.state_dict().items() if not torch.equal(p.cpu(),expected[n])]
        opt_equal=all(torch.equal(v.cpu(),optim_expected[k][a]) if torch.is_tensor(v) else v==optim_expected[k][a] for k,vs in opt.state_dict()['state'].items() for a,v in vs.items())
        row={'T':T,'weights_bitwise_equal':not unequal,'unequal_tensor_count':len(unequal),'max_weight_difference':max(float((p.cpu()-expected[n]).abs().max()) for n,p in m.state_dict().items()),
            'optimizer_bitwise_equal':opt_equal,'sampling_equal':torch.equal(g.get_state(),gen),'eager_seconds':eager_times,'graph_seconds':graph_times}
        rows.append(row);print(json.dumps(row),flush=True)
        del fn,m,opt,params;gc.collect();torch.cuda.empty_cache();init.unlink()
    save_json(DEST/'E1/training_graph_check.json',rows)

if __name__=='__main__':check()
