"""Freeze model/analysis list before evaluating untouched language test data."""
from paper_train import *
import argparse,hashlib,requests,concurrent.futures,subprocess

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def freeze():
    assert (DEST/'training_finished.json').exists()
    path=DEST/'confirm_manifest.json'
    if path.exists():return json.loads(path.read_text())
    rows=[{'id':f'frozen_T{T}','T':T,'seed':None,'stage':'frozen','step':0,'training_seconds':0,'checkpoint':None,'gpu':0} for T in [2,4]]
    formal=[]
    for p in sorted((DEST/'formal').glob('*/complete.json')):
        r=json.loads(p.read_text());c=r['config'];folder=Path(r['checkpoint']).parent
        formal.append((c,folder,r))
    assert len(formal)==6
    h=min(r['training_seconds'] for c,f,r in formal);time_rows=[]
    for c,folder,r in formal:
        progress=[json.loads(s) for s in (DEST/'formal'/folder.name/'steps.jsonl').read_text().splitlines()]
        checkpoint_times={s['step']:s['cumulative_training_seconds'] for s in progress if s['step']%256==0}
        steps=set([512,1024,2048,4096])
        for fraction in [.25,.5,1.]:
            threshold=h*fraction;eligible=[s for s,t in checkpoint_times.items() if t<=threshold]
            if not eligible:time_rows.append({'T':c['T'],'seed':c['seed'],'threshold':threshold,'fraction':fraction,'status':'no_checkpoint_before_threshold'});continue
            s=max(eligible);steps.add(s);slack=(threshold-checkpoint_times[s])/threshold
            time_rows.append({'T':c['T'],'seed':c['seed'],'threshold':threshold,'fraction':fraction,'step':s,'actual_seconds':checkpoint_times[s],
                              'slack_fraction':slack,'strict_time_match':slack<=.1,'gpu':c['visible_devices']})
        for s in sorted(steps):
            cp=folder/f'step{s}.pt';assert cp.exists()
            rows.append({'id':f'formal_T{c["T"]}_seed{c["seed"]}_step{s}','T':c['T'],'seed':c['seed'],'stage':'formal','step':s,
                         'training_seconds':checkpoint_times[s],'checkpoint':str(cp),'gpu':int(c['visible_devices']),'lr':c['lr']})
    for p in sorted((DEST/'legacy').glob('*/complete.json')):
        r=json.loads(p.read_text());c=r['config']
        rows.append({'id':f'legacy_reconstructed_T{c["T"]}_seed{c["seed"]}_step512','T':c['T'],'seed':c['seed'],'stage':'legacy_reconstructed',
                    'step':512,'training_seconds':r['training_seconds'],'checkpoint':r['checkpoint'],'gpu':int(c['visible_devices']),'lr':c['lr']})
    for row in rows:
        row['checkpoint_sha256']=digest(row['checkpoint'] or ROOT/'data/fused_state.pt')
    record={'locked_at':time.time(),'models':rows,'time_budget':{'h':h,'points':time_rows},'primary':'formal_4096_matched_and_unmatched_CE',
            'secondary':['legacy_reconstruction_512','fixed_update_curve','fixed_time_curve','masked_accuracy'],
            'splits':{'test':{'source':'Salesforce/wikitext/wikitext-2-raw-v1 official test','mask_seed':8702},'supplementary':{'order_range':[3616,3849],'mask_seed':8701,'independence':'104 blocks overlap historical contexts'}},
            'bootstrap_replicates':10000,'bootstrap_seed':8901,'analysis_code_sha256':digest(ROOT/'code/paper_analyze.py'),
            'confirmation_code_sha256':digest(__file__),'final_confirmation_evaluated_before_lock':False}
    save_json(path,record);save_json(DEST/'time_budget_checkpoints.json',record['time_budget']);return record

def prepare_confirmation():
    assert (DEST/'confirm_manifest.json').exists(),'Manifest must exist before loading final split'
    target=ROOT/'data/test.parquet';source_record=DEST/'E0/test_source.json'
    if not target.exists():
        errors=[]
        for host in ['https://huggingface.co','https://hf-mirror.com']:
            url=host+'/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet'
            try:
                r=requests.get(url,timeout=(15,90));r.raise_for_status();assert r.content[:4]==b'PAR1'
                target.write_bytes(r.content);save_json(source_record,{'url':url,'repo_commit':r.headers.get('X-Repo-Commit'),'etag':r.headers.get('ETag'),'sha256':digest(target),'bytes':len(r.content)});break
            except Exception as e:errors.append(str(e))
        if not target.exists():raise RuntimeError('Official test download failed: '+str(errors))
    import pyarrow.parquet as pq
    tok=BertTokenizerFast.from_pretrained(str(VENDOR/'tokenizer_files'),local_files_only=True)
    texts=pq.read_table(target,columns=['text']).to_pydict()['text'];ids=[]
    for start in range(0,len(texts),256):
        for row in tok(texts[start:start+256],add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)['input_ids']:ids.extend(row)
    n=len(ids)//62;middle=torch.tensor(ids[:n*62],dtype=torch.long).reshape(n,62)
    raw=torch.cat([torch.full((n,1),101),middle,torch.full((n,1),102)],1)
    x,y=mask_batch(raw,torch.Generator().manual_seed(8702))
    d=prepare_data();order=torch.randperm(len(d['validation']),generator=torch.Generator().manual_seed(6101));sids=order[3616:3849]
    sx,sy=mask_batch(d['validation'][sids],torch.Generator().manual_seed(8701))
    # Exact raw-block overlap audit; retain the full predefined official test.
    train_set={tuple(row) for row in d['train'].tolist()};validation_set={tuple(row) for row in d['validation'].tolist()}
    overlaps_train=[i for i,row in enumerate(raw.tolist()) if tuple(row) in train_set]
    overlaps_val=[i for i,row in enumerate(raw.tolist()) if tuple(row) in validation_set]
    save_json(DEST/'E0/test_overlap_audit.json',{'test_blocks':n,'exact_train_overlap_blocks':overlaps_train,'exact_validation_overlap_blocks':overlaps_val,
        'policy':'Full predefined test is reported; overlap identities logged before model scoring. No favorable subset selection.', 'upstream_pretraining_exposure':'unknown'})
    atomic_save({'test':{'x':x,'y':y,'ids':torch.arange(n)},'supplementary':{'x':sx,'y':sy,'ids':sids}},CACHE/'confirmation_inputs.pt')
    save_json(DEST/'confirmation_data.json',{'test':{'blocks':n,'masked_tokens':int((y!=-100).sum()),'mask_seed':8702},'supplementary':{'blocks':len(sids),'masked_tokens':int((sy!=-100).sum()),'mask_seed':8701}})

def evaluate_worker(gpu):
    manifest=json.loads((DEST/'confirm_manifest.json').read_text());data=torch.load(CACHE/'confirmation_inputs.pt',weights_only=True)
    for row in manifest['models']:
        if row['gpu']!=gpu:continue
        out=DEST/'confirmation'/row['id'];out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        m=model(row['T'],row['checkpoint'],train=False).eval();results={}
        for name,d in data.items():results[name]=assess(m,d['x'],d['y'],d['ids'],out/f'{name}.npz')
        save_json(out/'complete.json',{'model':row,'results':results});print(json.dumps({'model':row['id'],**results}),flush=True)
        del m;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','worker']);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='prepare':prepare_confirmation()
    else:evaluate_worker(a.gpu)
