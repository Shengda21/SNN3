"""Lock the finite scoring matrix after validation-only hyperparameter selection."""
import json,os,time
from pathlib import Path
from r2_vision import WORK,INIT,write_json
from r3_score import digest,LEGACY

def main():
    selection=json.loads((WORK/'results/selection.json').read_text())['selected']
    rows=[];identities={}
    def identity(path):
        key=str(path)
        if key not in identities:identities[key]=digest(path)
        return identities[key]
    def add(source_id,path,init_id,state,seed,a,K,role,full,lr=None):
        common={'task':'vision','checkpoint':str(path),'checkpoint_sha256':identity(path),'source_id':source_id,
                'init_id':init_id,'initialization_seed':12450 if init_id=='A' else 12550,
                'state':state,'seed':seed,'adapt_T':a,'lr':lr,'updates':K,'role':role,'engine':'graph'}
        for e in [2,4]:rows.append({**common,'id':f'{source_id}_e{e}_raw','infer_T':e,'calibration':'raw'})
        for subset in ([9640,9740,9840] if full else [9640]):
            for c in [2,4]:
                for e in ([2,4] if full else [c]):
                    rows.append({**common,'id':f'{source_id}_c{c}_e{e}_bn{subset}',
                                 'infer_T':e,'cal_T':c,'cal_seed':subset,'calibration':'bn_cross'})
    for init_id,seeds,stage,initial in [
        ('A',range(9621,9626),'formal',INIT),
        ('B',range(9721,9724),'formal_B',WORK/'initialization/results/independent12550/best.pt')]:
        add(f'{init_id}_initial',initial,init_id,'shared_initial',None,None,0,'primary',True)
        for seed in seeds:
            for T in [2,4]:
                lr=selection[str(T)];tag=f's{seed}_T{T}_lr{lr:g}'
                record=json.loads((WORK/'results'/stage/tag/'complete.json').read_text())
                assert record['step']==2048 and record['status']=='complete'
                for K in [512,1024,2048]:
                    add(f'{init_id}_s{seed}_a{T}_K{K}',WORK/'models'/stage/tag/f'step{K}.pt',init_id,'adapted',seed,T,K,'primary',K==2048,lr)
                if init_id=='A' and lr!=2e-5:
                    tag=f's{seed}_T{T}_lr2e-05'
                    add(f'{init_id}_common_s{seed}_a{T}_K2048',WORK/'models'/stage/tag/'step2048.pt',init_id,'adapted',seed,T,2048,'common_lr',False,2e-5)
    # All retained R2.1 outputs are replayed under the new device/runtime.
    legacy=json.loads((LEGACY/'results/model_analysis_lock.json').read_text())
    for source in legacy['scoring']:
        r=dict(source)
        assert identity(r['identity_path'])==r['checkpoint_sha256']
        r.update(id='legacy_'+r['id'],legacy_id=r['id'],role='legacy_replay',init_id='legacy')
        if r['calibration']=='bn_only':r.update(cal_T=r['infer_T'],cal_seed=9640)
        rows.append(r)
    assert len(rows)==len({r['id'] for r in rows})
    write_json(WORK/'protocol/scoring_queue.json',rows)
    lock={'version':'R3','locked_at':time.time(),'selected':selection,'source_identities':identities,
          'rows':len(rows),'new_scores_observed_before_lock':False,'legacy_results_previously_observed':True,
          'source_files':{p.name:digest(p) for p in (WORK/'code').glob('*.py')}}
    write_json(WORK/'results/model_analysis_lock.json',lock)
    print(json.dumps({'rows':len(rows),'checkpoints':len(identities)}),flush=True)
if __name__=='__main__':main()
