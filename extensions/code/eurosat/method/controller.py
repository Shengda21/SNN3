import sys as _blood_sys
from pathlib import Path as _BloodPath
_blood_sys.path.insert(0,str(_BloodPath(__file__).resolve().parents[1]))
"""Finite measured campaign: constructor terminal -> development -> gate -> formal -> test."""
import argparse,importlib.util,itertools,json,os,time,traceback
from pathlib import Path
from port_common import *

class Campaign:
    def __init__(self,plan):
        self.p=plan;self.out=Path(plan['output']);self.bundle=Path(plan['constructor_bundle'])
        self.parent=Path(plan['construction_parent']);self.records={};self.stages=[]
    def snapshot(self,stage,**extra):
        value={'PID':os.getpid(),'stage':stage,'unix':time.time(),'source_freeze_sha256':source_hash(),**extra}
        path=self.out/'status.json';tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)
    def wait_construction(self):
        deadline=time.monotonic()+self.p['constructor_wait_deadline_seconds']
        roles=('development','formal1','formal2','formal3')
        while True:
            for failure in self.p['construction_failure_files']:
                f=Path(failure)
                require(not f.exists() or f.read_text().strip()=='0','constructor queue failed; preserve and stop: '+str(f))
            ready=all((self.parent/role/'construction/C-conf/complete.json').is_file() for role in roles)
            ready=ready and all(Path(f).is_file() and Path(f).read_text().strip()=='0' for f in self.p['construction_failure_files'])
            transfer_terminal=Path(self.p['construction_transfer_terminal'])
            ready=ready and transfer_terminal.is_file()
            if ready:
                transfer=read(transfer_terminal)
                require(transfer['schema']=='eurosat64_constructor_manifest_v1' and transfer['status']=='complete','actual completed samehost frozen constructor snapshot')
                imported={x['role']:x for x in transfer['roles']}
                require(set(imported)==set(roles[1:]),'exact independent formal roles transferred')
                terminals=[]
                for role in roles:
                    root=self.parent/role
                    verify_parent(self.bundle,root)
                    verify_checkpoint(root,root/'construction/C-conf/best.pt','S0')
                    terminal=root/'construction/C-conf/complete.json'
                    row={'role':role,'path':str(terminal),'sha256':sha(terminal)}
                    if role!='development':
                        receipt=imported[role]
                        require(receipt['root']==str(root) and receipt['complete_sha256']==sha(terminal),'exact completed own-host constructor provenance')
                        manifest=Path(receipt['manifest_path'])
                        require(sha(manifest)==receipt['manifest_sha256'],'actual source transfer manifest bytes')
                        files=read(manifest);require(len(files)==receipt['files_count'] and bool(files),'complete source tree transfer manifest')
                        for f in files:
                            path=root/f['path'];require(path.is_file() and sha(path)==f['sha256'] and path.stat().st_size==f['size'],'transferred source bytes '+str(path))
                        row.update(source_server=receipt['source_server'],transfer_manifest=str(manifest),transfer_manifest_sha256=sha(manifest))
                    terminals.append(row)
                verified=self.out/'constructor_verified_roles.json'
                write(verified,{'status':'complete','roles':terminals,'transfer_terminal':str(transfer_terminal),'transfer_terminal_sha256':sha(transfer_terminal)})
                write(self.out/'constructor_terminal.json',{'path':str(verified),'sha256':sha(verified)});return
            require(time.monotonic()<deadline,'finite constructor wait expired; no automatic continuation')
            self.snapshot('waiting_actual_constructor_and_transfer_terminals',roles={r:str(self.parent/r/'construction/C-conf/complete.json') for r in roles},transfer_terminal=str(transfer_terminal))
            time.sleep(20)
    def stage(self,name,script,argv,terminal,wait=True):
        from distributed_spool import submit,collect
        request=submit(self.p,self.out,name,script,list(map(str,argv)),str(terminal));self.snapshot(name,request=request['id'])
        if not hasattr(self,'pending'):self.pending=[]
        if wait:self.stages.append(collect(self.p,self.out,request))
        else:self.pending.append(request)
    def grid_barrier(self):
        from distributed_spool import collect
        for request in self.pending:self.stages.append(collect(self.p,self.out,request))
        self.pending=[]
    def native(self,role,seed,method,rate,index,phase,wait=True):
        root=self.parent/role;ident=f'{phase}_{method}_s{seed}_r{index:02d}';dest=self.out/'trajectories'/ident
        initial=root/'construction/C-conf/best.pt';job={'id':ident,'architecture':'resformer','cohort':'EuroSAT_RGB-Resformer',
             'method':'mtt' if method=='mtt' else 'native','T':None if method=='mtt' else int(method[-1]),
             'seed':seed,'lr':rate,'steps':2048,'budgets':list(BUDGETS),'initial_checkpoint':str(initial),
             'initial_checkpoint_sha256':sha(initial)}
        if phase=='formal':
            job['selection_authority']={'path':str(self.out/'rate_selection.json'),'sha256':sha(self.out/'rate_selection.json')}
            job['development_gate_authority']={'path':str(self.out/'development_gate.json'),'sha256':sha(self.out/'development_gate.json')}
        path=self.out/'jobs'/(ident+'.json');write(path,job)
        terminal=dest/'training'/ident/'complete.json'
        self.stage(ident,'native.py',['--bundle',self.bundle,'--construction-root',root,'--job',path,'--out',dest],terminal,wait=(wait and phase!='screen'))
        record={'id':ident,'root':root,'output':dest,'terminal':terminal,'seed':seed,'method':method,'rate':rate,
                'checkpoints':{k:dest/'models'/ident/f'step{k}.pt' for k in BUDGETS}}
        self.records[ident]=record;return record
    def conditional(self,role,seed,acceptance=False,wait=True):
        root=self.parent/role;name=f'{role}_conditional_s{seed}'+('_acceptance' if acceptance else '')
        dest=self.out/'trajectories'/name
        args=['--bundle',self.bundle,'--construction-root',root,'--seed',seed,'--out',dest]
        if acceptance:args+=['--acceptance-only']
        elif role!='development':args+=['--development-gate',self.out/'development_gate.json']
        self.stage(name,'train.py',args,dest/'complete.json',wait=(wait or acceptance))
        if acceptance:return
        fit=self.out/'fits'/name
        self.stage(name+'_fit','fit_pack.py',['--bundle',self.bundle,'--construction-root',root,'--training',dest,'--out',fit],fit/'complete.json',wait=wait)
        record={'id':name,'root':root,'seed':seed,'method':'conditional','checkpoints':{k:fit/f'final_step{k}.pt' for k in BUDGETS}}
        self.records[name]=record;return record
    def score(self,record,k,t,b,split='validation',wait=True):
        name=f'{split}_{record["id"]}_k{k}_t{t}_{b}';dest=self.out/'scores'/name;terminal=dest/'complete.json'
        if name in self.scored:return self.scored[name]
        method='conditional' if record['method']=='conditional' else 'S0' if record['method']=='S0' else 'native'
        args=['--bundle',self.bundle,'--construction-root',record['root'],'--checkpoint',record['checkpoints'][k],
              '--method',method,'--window',t,'--policy',b,'--split',split,'--out',dest]
        if split=='test':args+=['--test-freeze',self.out/'FINAL_TEST_FREEZE.json']
        self.stage(name,'score.py',args,terminal,wait=wait);self.scored[name]=terminal;return terminal
    def prefetch(self,records,split='validation'):
        for record in records:
            for k,t,b in itertools.product((0,) if record['method']=='S0' else BUDGETS,(2,4),('raw','matched')):self.score(record,k,t,b,split,wait=False)
    def matrix(self,unit,s0,reuse,candidate,others=(),split='validation'):
        rows=[]
        for record in (s0,reuse,candidate,*others):
            ks=(0,) if record['method']=='S0' else BUDGETS
            for k,t,b in itertools.product(ks,(2,4),('raw','matched')):
                path=self.score(record,k,t,b,split)
                if record in (s0,reuse,candidate):
                    method='S0' if record is s0 else 'reuse2' if record is reuse else 'conditional'
                    rows.append({**read(path),'unit':unit,'method':method,'K':k,'score_source':str(path),'score_source_sha256':sha(path)})
        return rows
    def s0(self,role):
        root=self.parent/role;return {'id':role+'_S0','root':root,'method':'S0','checkpoints':{0:root/'construction/C-conf/best.pt'}}
    def selection(self,screen):
        choices={};evidence=[]
        for method in ('native2','native4','mtt'):
            values=[]
            for rate in self.p['mtt_rates' if method=='mtt' else 'native_rates']:
                scores=[]
                for r in [r for r in screen if r['method']==method and r['rate']==rate]:
                    require(read(r['terminal'])['status']=='complete','full selected screen trajectory')
                    if method=='mtt':
                        cells=[self.score(r,2048,t,'matched') for t in (2,4)]
                        scores.append(sum(read(p)['ordinary_CE'] for p in cells)/2)
                        evidence.extend({'path':str(p),'sha256':sha(p)} for p in cells)
                    else:
                        terminal=read(r['terminal']);evals=terminal['evaluations'];last=next(e for e in evals if e['step']==2048)
                        scores.append(last['raw_validation_by_window'][method[-1]]['ce'])
                        evidence.append({'path':str(r['terminal']),'sha256':sha(r['terminal'])})
                require(len(scores)==2,'both fixed development streams at every original rate')
                values.append({'rate':rate,'mean_validation_CE':sum(scores)/2,'paired_stream_CE':scores})
            winner=min(values,key=lambda v:(v['mean_validation_CE'],v['rate']))
            choices[method]={'rate':winner['rate'],'grid':values}
        write(self.out/'rate_selection.json',{'status':'complete','schema':SCHEMA,'source_freeze_sha256':source_hash(),
              'construction_root':str(self.parent/'development'),'paired_seeds':[34611,34612],'choices':choices,
              'selection_uses_test':False,'evidence':evidence})
        return choices
    def run(self):
        self.wait_construction()
        (self.out/'measurement').mkdir();self.scored={}
        self.conditional('development',34611,acceptance=True)
        screen=[]
        for method in ('native2','native4','mtt'):
            rates=self.p['mtt_rates' if method=='mtt' else 'native_rates']
            for i,rate in enumerate(rates):
                for seed in self.p['development_seeds']:screen.append(self.native('development',seed,method,rate,i,'screen'))
        self.grid_barrier();choices=self.selection(screen);devrows=[]
        candidates={seed:self.conditional('development',seed,wait=False) for seed in self.p['development_seeds']}
        self.grid_barrier()
        selected_by_seed={seed:{m:next(r for r in screen if r['method']==m and r['seed']==seed and r['rate']==choices[m]['rate']) for m in choices} for seed in self.p['development_seeds']}
        for seed in self.p['development_seeds']:
            self.prefetch([self.s0('development'),candidates[seed],*selected_by_seed[seed].values()])
        self.grid_barrier()
        for seed in self.p['development_seeds']:
            selected=selected_by_seed[seed];devrows+=self.matrix(f'dev{seed}',self.s0('development'),selected['native2'],candidates[seed],(selected['native4'],selected['mtt']))
        rows=self.out/'development_rows.json';write(rows,devrows)
        from gate import bound_assess
        gate=bound_assess(rows,['dev34611','dev34612'],'development',self.parent/'development');write(self.out/'development_gate.json',gate)
        if not gate['gate_passed']:
            self.audit_scores('development_negative')
            write(self.out/'negative_terminal.json',{'status':'scientific_negative_development_gate','gate_sha256':sha(self.out/'development_gate.json'),
                  'analysis_summary_sha256':sha(self.out/'analysis/summary.json'),
                  'all_score_receipts_sha256':sha(self.out/'all_score_receipts.json'),
                  'no_method_change':True,'no_threshold_change':True,'test_read':False})
            self.handoff('scientific_negative_development_gate',self.out/'negative_terminal.json');return
        formal=[];formalrows=[]
        for role,seed in self.p['formal_role_seeds'].items():
            baselines={m:self.native(role,seed,m,choices[m]['rate'],0,'formal',wait=False) for m in choices}
            candidate=self.conditional(role,seed,wait=False);s0=self.s0(role);formal.append((f'formal{seed}',s0,baselines,candidate))
        self.grid_barrier()
        for unit,s0,baselines,candidate in formal:self.prefetch([s0,candidate,*baselines.values()])
        self.grid_barrier()
        for unit,s0,baselines,candidate in formal:formalrows+=self.matrix(unit,s0,baselines['native2'],candidate,(baselines['native4'],baselines['mtt']))
        rows=self.out/'formal_rows.json';write(rows,formalrows)
        formal_gate=bound_assess(rows,['formal34621','formal34622','formal34623'],'formal',self.parent)
        write(self.out/'formal_gate.json',formal_gate)
        if not formal_gate['gate_passed']:
            self.audit_scores('formal_negative')
            write(self.out/'negative_terminal.json',{'status':'scientific_negative_formal_gate','gate_sha256':sha(self.out/'formal_gate.json'),
                  'analysis_summary_sha256':sha(self.out/'analysis/summary.json'),
                  'all_score_receipts_sha256':sha(self.out/'all_score_receipts.json'),
                  'no_method_change':True,'no_threshold_change':True,'test_read':False})
            self.handoff('scientific_negative_formal_gate',self.out/'negative_terminal.json');return
        checkpoints=[]
        for _,s0,baselines,candidate in formal:
            for r in (s0,*baselines.values(),candidate):
                method='S0' if r['method']=='S0' else 'conditional' if r['method']=='conditional' else 'native'
                checkpoints.extend({'path':str(p),'sha256':sha(p),'method':method,'construction_root':str(r['root'])} for p in r['checkpoints'].values())
        write(self.out/'FINAL_TEST_FREEZE.json',{'schema':'eurosat64_final_test_freeze_v1','selection_complete':True,'checkpoints':checkpoints,
              'method_source_freeze_sha256':source_hash(),'rate_selection_sha256':sha(self.out/'rate_selection.json'),
              'development_gate_sha256':sha(self.out/'development_gate.json'),
              'formal_gate_sha256':sha(self.out/'formal_gate.json'),'test_read_before_freeze':False})
        # Both original development and independent formal gates passed before any test entry.
        for unit,s0,baselines,candidate in formal:self.prefetch([s0,candidate,*baselines.values()],split='test')
        self.grid_barrier()
        for unit,s0,baselines,candidate in formal:self.matrix(unit,s0,baselines['native2'],candidate,(baselines['native4'],baselines['mtt']),split='test')
        self.audit_scores('scientific_complete')
        write(self.out/'complete.json',{'status':'complete','source_freeze_sha256':source_hash(),
               'stages':self.stages,'score_routes':len(self.scored),'rate_selection_sha256':sha(self.out/'rate_selection.json'),
               'formal_gate_sha256':sha(self.out/'formal_gate.json'),'all_score_receipts_sha256':sha(self.out/'all_score_receipts.json'),
               'analysis_summary_sha256':sha(self.out/'analysis/summary.json')})
        self.handoff('scientific_complete',self.out/'complete.json')
    def audit_scores(self,outcome):
        summaries=[{'path':str(path),'sha256':sha(path),'row':read(path)} for path in self.scored.values()]
        write(self.out/'all_score_receipts.json',summaries)
        from analyze import analyze
        analyze(self.out/'all_score_receipts.json',self.out/'analysis',outcome=outcome)
    def handoff(self,outcome,terminal):
        write(self.out/'handoff_terminal.json',{'schema':'eurosat64_finite_handoff_v1_2','status':'complete','outcome':outcome,
              'scientific_gate_passed':outcome=='scientific_complete',
              'scientific_terminal':{'path':str(terminal),'sha256':sha(terminal)},
              'method_schema':SCHEMA,'source_freeze_sha256':source_hash(),
              'plan_sha256':sha(self.out/'plan.json'),'constructor_terminal':read(self.out/'constructor_terminal.json'),
              'stages':self.stages,'automatic_retry':False})

def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,default=HERE/'campaign_plan.json');a.add_argument('--execute',action='store_true');args=a.parse_args()
    p=read(args.plan);require(p['schema']=='eurosat64_finite_campaign_20261030_v1_2' and p['physical_gpu']==0 and p['selection_uses_test'] is False
            and p['automatic_retry'] is False and p['budgets']==list(BUDGETS),'frozen finite campaign scope')
    require(p==read(HERE/'campaign_plan.json'),'exact reviewed finite campaign plan; no runtime protocol replacement')
    require(p['development_seeds']==[34611,34612] and p['formal_role_seeds']=={'formal1':34621,'formal2':34622,'formal3':34623},'all paired starts frozen')
    if args.execute:
        from authority import require_production
        require_production()
    if not args.execute:print(json.dumps({'status':'validated_only','screen_trajectories':38,'formal_baseline_trajectories':9,'conditional_scientific_trajectories':5,
        'constructor_dependency':p['constructor_queue'],'finite_failure_action':p['failure_action']}));return
    verify_sources();campaign=Campaign(p);campaign.out.mkdir(parents=True,exist_ok=False);write(campaign.out/'plan.json',p)
    write(campaign.out/'started.json',{'PID':os.getpid(),'source_freeze_sha256':source_hash(),'plan_sha256':sha(args.plan),'unix':time.time()})
    try:campaign.run()
    except BaseException:
        write(campaign.out/'failed.json',{'status':'failed_preserved','automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
