"""Fixed quarter-tail means. Never installs a mean into the training model."""
import copy

RANGES = {512:(385,512),1024:(769,1024),2048:(1537,2048)}

class TailAverage:
    def __init__(self, parameter_names):
        self.names=tuple(parameter_names);self.total={};self.count=0;self.last_step=128;self.last_sum=None

    def committed(self, step, state):
        if step != self.last_step+1:raise RuntimeError('nonconsecutive successful attempt')
        self.last_step=step
        active=next((k for k,(a,b) in RANGES.items() if a<=step<=b),None)
        if active is None:return None
        if not self.total:self.total={n:state[n].detach().double().clone() for n in self.names}
        else:
            for n in self.names:self.total[n].add_(state[n].detach().double())
        self.count+=1
        a,b=RANGES[active]
        if step!=b:return None
        if self.count!=b-a+1:raise RuntimeError('tail count differs')
        # Preserve raw references/counters without averaging; only named parameters.
        result=copy.deepcopy(state)
        for n in self.names:result[n]=(self.total[n]/self.count).to(dtype=state[n].dtype)
        receipt={'range_inclusive':[a,b],'count':self.count,'arithmetic':'FP64 sum/count then original dtype'}
        self.last_sum={n:v.detach().cpu().clone() for n,v in self.total.items()}
        self.total={};self.count=0
        return result,receipt
