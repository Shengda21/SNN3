"""Evidence-bound summaries; no filtering to successful scientific hypotheses."""
import csv,json,math,time,re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/paper_experiments_20260908'

def write_json(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def difference(a,b):
    # Rows are paired training seeds; columns are identical scored text blocks.
    assert len(a)==len(b)
    counts=a[0]['counts'];ids=a[0]['block_id']
    assert all(np.array_equal(d['counts'],counts) and np.array_equal(d['block_id'],ids) for d in a+b)
    diff=np.stack([x['loss_sum']-y['loss_sum'] for x,y in zip(a,b)])
    accdiff=np.stack([x['correct']-y['correct'] for x,y in zip(a,b)])
    each=diff.sum(1)/counts.sum();acceach=100*accdiff.sum(1)/counts.sum()
    rng=np.random.default_rng(8901);boot=[]
    for _ in range(100):
        ix=rng.integers(0,len(counts),(100,len(counts)));boot.extend((diff.mean(0)[ix].sum(1)/counts[ix].sum(1)).tolist())
    cluster_ids=np.unique(ids//8);cluster_diff=np.array([diff.mean(0)[ids//8==c].sum() for c in cluster_ids]);cluster_count=np.array([counts[ids//8==c].sum() for c in cluster_ids])
    rng=np.random.default_rng(8901);cluster_boot=[]
    for _ in range(100):
        ix=rng.integers(0,len(cluster_ids),(100,len(cluster_ids)));cluster_boot.extend((cluster_diff[ix].sum(1)/cluster_count[ix].sum(1)).tolist())
    sd=float(each.std(ddof=1));se=sd/math.sqrt(len(each));radius=float(student_t.ppf(.975,len(each)-1))*se
    ci=np.quantile(boot,[.025,.975]).tolist();cci=np.quantile(cluster_boot,[.025,.975]).tolist()
    return {'ce_difference_mean':float(each.mean()),'ce_difference_each_seed':each.tolist(),'ce_difference_sd':sd,
        'conditional_block_ci95':ci,'original_order_8block_cluster_ci95':cci,'paired_seed_t_ci95':[float(each.mean()-radius),float(each.mean()+radius)],
        'accuracy_difference_pp_each':acceach.tolist(),'accuracy_difference_pp_mean':float(acceach.mean()),
        'all_seeds_positive':bool((each>0).all()),'all_seeds_negative':bool((each<0).all()),
        'stable_positive_conditional':bool((each>0).all() and ci[0]>0),'stable_negative_conditional':bool((each<0).all() and ci[1]<0)}

def main():
    manifest=json.loads((OUT/'confirm_manifest.json').read_text());rows=[];arrays={};byid={}
    for m in manifest['models']:
        p=OUT/'confirmation'/m['id'];r=json.loads((p/'complete.json').read_text());byid[m['id']]=r
        for split,score in r['results'].items():
            rows.append({**{k:v for k,v in m.items() if k not in ['checkpoint_sha256']},'split':split,**score,'accuracy_pct':100*score['accuracy']})
            arrays[m['id'],split]=np.load(p/f'{split}.npz')
    write_csv('quality.csv',rows);differences=[]
    for split in ['test','supplementary']:
        for stage,seeds,steps in [('formal',[8621,8622,8623],[512,1024,2048,4096]),('legacy_reconstructed',[7621,7622,7623],[512])]:
            for step in steps:
                try:
                    a=[arrays[f'{stage}_T2_seed{s}_step{step}',split] for s in seeds]
                    b=[arrays[f'{stage}_T4_seed{s}_step{step}',split] for s in seeds]
                except KeyError:continue
                for name,c in [('matched',b),('unmatched',[arrays['frozen_T4',split]]*3)]:
                    differences.append({'split':split,'stage':stage,'step':step,'comparison':name,**difference(a,c)})
        times=manifest['time_budget']['points']
        for fraction in [.25,.5,1.]:
            choices={(r['T'],r['seed']):r for r in times if r['fraction']==fraction and 'step' in r}
            if len(choices)!=6:continue
            a=[arrays[f'formal_T2_seed{s}_step{choices[2,s]["step"]}',split] for s in [8621,8622,8623]]
            b=[arrays[f'formal_T4_seed{s}_step{choices[4,s]["step"]}',split] for s in [8621,8622,8623]]
            differences.append({'split':split,'stage':'formal_equal_time','fraction':fraction,'comparison':'matched',
                'strict_time_match_all':all(v['strict_time_match'] for v in choices.values()),**difference(a,b)})
    write_json('quality_differences.json',differences);write_csv('quality_differences.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in d.items()} for d in differences])
    latency=[];units=[];failed=[]
    for p in (OUT/'benchmark').rglob('T*_b*.json'):
        if p.parents[1].name not in ['historical','formal','legacy','masked']:continue
        r=json.loads(p.read_text())
        if r.get('status')!='complete':failed.append({'path':str(p),**r});continue
        base={k:r.get(k) for k in ['architecture','checkpoint','tag','T','batch','route','session','gpu','gpu_id','torch','cuda','copy_scope','warmup_capture_seconds']}
        for call in r['measurements']:latency.append({**base,**call})
        units.append({**base,'group':p.parents[1].name,'medians':r['medians'],'all_checks_bitwise':all(c['bitwise_equal'] for c in r['checks'])})
    write_csv('latency.csv',latency);write_json('latency_units.json',units);write_json('benchmark_unavailable.json',failed)
    pairs=[]
    def pairtag(r):return re.sub(r'_T[24]_', '_Tpaired_',r['tag']) if r['group'] in ['formal','legacy'] else r['tag']
    keys={(r['architecture'],pairtag(r),r['session'],r['batch'],r['route'],r['group']) for r in units}
    for arch,tag,session,batch,route,group in sorted(keys):
        found={r['T']:r for r in units if (r['architecture'],pairtag(r),r['session'],r['batch'],r['route'],r['group'])==(arch,tag,session,batch,route,group)}
        if set(found)!=set([2,4]):continue
        a=found[2]['medians'];b=found[4]['medians']
        pair={'architecture':arch,'tag':tag,'session':session,'batch':batch,'route':route,'group':group,
              'cross_T2_eager_over_T4_graph':a['eager']/b['graph'],'same_T4_graph_over_T2_graph':b['graph']/a['graph']}
        if 'eager_with_copy' in a:pair['matched_copy_T2_eager_over_T4_graph']=a['eager_with_copy']/b['graph_with_copy']
        pairs.append(pair)
    write_json('execution_ratios.json',pairs)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def savefig(name,fig):
        for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,split in zip(axs,['test','supplementary']):
        for T,color in [(2,'#2478a8'),(4,'#bd5c38')]:
            steps=[512,1024,2048,4096];values=np.array([[r['ce'] for r in rows if r['stage']=='formal' and r['T']==T and r['step']==s and r['split']==split] for s in steps])
            ax.errorbar(steps,values.mean(1),yerr=values.std(1,ddof=1),marker='o',color=color,label=f'T={T}',capsize=3)
        ax.set(xlabel='Updates',ylabel='Masked-token CE',title=split);ax.set_xscale('log',base=2);ax.legend(frameon=False)
    savefig('quality_budget',fig)
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    for T,color in [(2,'#2478a8'),(4,'#bd5c38')]:
        val=[]
        for seed in [8621,8622,8623]:
            p=next((OUT/'formal').glob(f'T{T}_seed{seed}_*/evaluations.json'));es=json.loads(p.read_text());val.append([e['tune']['ce'] for e in es])
        arr=np.array(val);ax.errorbar([512,1024,2048,4096],arr.mean(0),yerr=arr.std(0,ddof=1),marker='o',color=color,label=f'T={T}')
    ax.set(xlabel='Updates',ylabel='Train-holdout CE',xscale='log');ax.legend(frameon=False);savefig('tune_budget',fig)
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    for session in [8801,8802,8803]:
        ps=sorted([r for r in pairs if r['architecture']=='language' and r['group']=='historical' and r['session']==session],key=lambda r:r['batch'])
        if ps:ax.plot([r['batch'] for r in ps],[r['cross_T2_eager_over_T4_graph'] for r in ps],marker='o',label=f'Session {session}')
    ax.axhline(1,color='gray',linestyle='--');ax.set(xscale='log',xlabel='Batch',ylabel='T2 eager / T4 graph');ax.legend(frameon=False);savefig('batch_boundary',fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    linked=[]
    for ax,batch in zip(axs,[1,4]):
        for r in units:
            if r['group']!='formal' or r['batch']!=batch:continue
            mid=r['tag'];q=byid.get(mid)
            if not q:continue
            for engine in ['eager','graph']:
                ce=q['results']['test']['ce'];ms=r['medians'][engine]
                ax.scatter(ms,ce,c='#2478a8' if r['T']==2 else '#bd5c38',marker='o' if engine=='eager' else 's')
                linked.append({'checkpoint_id':mid,'T':r['T'],'batch':batch,'engine':engine,'wall_ms':ms,'test_ce':ce,'gpu_id':r['gpu_id']})
        ax.set(xlabel='Own-checkpoint latency (ms)',ylabel='Official test CE',title=f'Batch {batch}')
    savefig('quality_latency',fig);write_csv('quality_latency.csv',linked)
    primary=[r for r in differences if r['split']=='test' and r['stage']=='formal' and r.get('step')==4096]
    mp=next(r for r in primary if r['comparison']=='matched');up=next(r for r in primary if r['comparison']=='unmatched')
    reversal=up['ce_difference_mean']<0 and mp['ce_difference_mean']>0
    hratios=[r for r in pairs if r['group']=='historical' and r['architecture']=='language' and r['batch']==1]
    lines=['# 新云环境实验报告','',f'正式4096步在官方test上的质量排序反转：{reversal}。',
           f'T2−匹配T4的CE差：{mp["ce_difference_mean"]:+.6f}；T2−冻结T4：{up["ce_difference_mean"]:+.6f}。',
           '','所有预定seed和候选均保留。下表的文本区间条件于三条已训练seed，不是跨训练总体区间。','',
           '|比较|平均CE差|三个seed差|条件文本95%区间|8块聚类95%区间|','|---|---:|---|---|---|']
    for r in primary:lines.append(f'|{r["comparison"]}|{r["ce_difference_mean"]:+.6f}|{r["ce_difference_each_seed"]}|{r["conditional_block_ci95"]}|{r["original_order_8block_cluster_ci95"]}|')
    lines+=['',f'语言冻结模型batch1三会话T2 eager/T4 graph：{[r["cross_T2_eager_over_T4_graph"] for r in hratios]}。',
            f'对应同引擎T4 graph/T2 graph：{[r["same_T4_graph_over_T2_graph"] for r in hratios]}。','',
            '硬件为RTX4090 D、每卡报告48GB；运行时、输入复制口径及逐调用记录见latency.csv。4096更新不代表充分收敛。',
            '233块补充数据存在104块历史上下文重合，不与官方test合并为独立测试。原模型跨环境复现差异见E0记录。',
            'legacy_reconstructed为重新训练的历史配置，不冒充找回的旧checkpoint。视觉分支资产和执行状态见vision_status.json。',
            '','产物：quality.csv、quality_differences.json、latency.csv、execution_ratios.json、quality_latency.csv、figures/。']
    (OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    claims={'RQ1':{'observed_reversal':reversal,'matched':mp,'unmatched':up},'RQ2':'Scope limited to declared learning-rate grid and 4096 updates; consult full budget curves.',
            'RQ3':'Equal-time rows include actual checkpoint slack and GPU assignment; approximate points labeled.',
            'RQ4':{'language_batch1_cross_ratios':[r['cross_T2_eager_over_T4_graph'] for r in hratios],'unavailable_units':len(failed)},
            'RQ5':'Only same-checkpoint measured latency joins in quality_latency.csv.'}
    write_json('claim_ledger.json',claims)
    (OUT/'PAPER_HANDOFF.md').write_text('# 论文写作交接\n\n从EXPERIMENT_REPORT.md与claim_ledger.json提取实际结论；逐项查证原始文件。更新近邻文献后再组织摘要、引言、控制协议、结果、局限与复现材料。不能把标准分解或CUDA Graph称为新算法。先解决任何vision_status或benchmark_unavailable中的缺失，再决定论文范围。\n',encoding='utf-8')
    write_json('analysis_complete.json',{'models_evaluated':len(manifest['models']),'quality_rows':len(rows),'timing_calls':len(latency),'joined_quality_latency_rows':len(linked),'time':time.time()})
    print(json.dumps({'primary_reversal':reversal,'matched_delta':mp['ce_difference_mean'],'unmatched_delta':up['ce_difference_mean'],'quality_rows':len(rows),'timing_calls':len(latency)}),flush=True)

if __name__=='__main__':main()
