"""Reconstruct an outcome-independent clean-train/val manifest; originals are retained."""
import argparse,collections,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def canonical(rows):return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def run(root):
    root=Path(root);source=root/'official_split_draft_v1.json';m=json.loads(source.read_text());rows=m['records'];by={r['path']:r for r in rows}
    near_path=root/'near_duplicate_audit.json';near=json.loads(near_path.read_text());rank={'train':0,'validation':1,'test':2}
    removed={};conflicts=[];internal=collections.Counter();crossings=collections.Counter()
    for group in near['nontrivial_components']:
        splits={by[p]['split'] for p in group};highest=max(rank[s] for s in splits)
        if len(splits)>1:
            crossings['+'.join(sorted(splits))]+=1
            for path in group:
                if rank[by[path]['split']]<highest:removed[path]={'reason':'component_overlaps_official_test' if highest==2 else 'component_overlaps_official_validation','component_paths':group}
        else:internal[next(iter(splits))]+=1
        labels={by[p]['label'] for p in group}
        if len(labels)>1:
            conflicts.append({'paths':group,'labels':sorted(labels),'class_names':sorted({by[p]['class_name'] for p in group}),
                              'official_splits':sorted(splits),'interpretation':'Source-label conflict/perceptual candidate; no outcome-based relabeling or filtering.'})
    clean=[r for r in rows if r['path'] not in removed];counts=collections.Counter(r['split'] for r in clean)
    assert dict(counts)=={'train':12741,'test':3677,'validation':1787},'Review changed counts; do not force an expected result'
    original_test=[r for r in rows if r['split']=='test'];clean_test=[r for r in clean if r['split']=='test']
    assert original_test==clean_test
    table=[]
    for label,name in enumerate(m['classes']):
        original=collections.Counter(r['split'] for r in rows if r['label']==label)
        kept=collections.Counter(r['split'] for r in clean if r['label']==label)
        table.append({'label':label,'class_name':name,'original':{s:original[s] for s in rank},
                      'kept':{s:kept[s] for s in rank},'removed':{s:original[s]-kept[s] for s in rank}})
    empty={s:[r['class_name'] for r in table if r['kept'][s]==0] for s in rank}
    small=[r for r in table if r['kept']['train']<50 or r['kept']['validation']<10 or r['kept']['test']<10]
    report={'status':'data_selection_audit','policy':'Fixed exact+existing perceptual connected components; test>validation>train; no threshold changes, webpage URL grouping, model outputs, or test performance used.',
            'original_counts':m['actual_counts'],'clean_counts':dict(counts),'removed_counts':dict(collections.Counter(by[p]['split'] for p in removed)),
            'total_excluded':len(removed),'test_records_byte_identity_in_canonical_manifest':True,'test_records_canonical_sha256':canonical(clean_test),
            'train_at_least2048':counts['train']>=2048,'empty_classes':empty,
            'small_class_reporting_thresholds_only':{'train_less_than':50,'validation_less_than':10,'test_less_than':10},
            'small_classes':small,'class_counts':table,'source_crossings':dict(crossings),'source_internal_duplicate_components':dict(internal),
            'cross_label_candidate_components':len(conflicts),'cross_label_groups':conflicts,
            'test_internal_duplicates_not_independent_samples':True,'source_membership_sha256':sha(source),'near_audit_sha256':sha(near_path),
            'archive_sha256':json.loads((root/'archive_verified.json').read_text())['sha256']}
    save(root/'clean_selection_audit_v1_1.json',report)
    proposal={**m,'dataset':'PlantWild_v1_derived_clean_train_val','records':clean,'actual_counts':dict(counts),
              'class_counts':{s:{r['class_name']:r['kept'][s] for r in table} for s in rank},
              'official_split_preserved':False,'official_test_preserved':True,'new_validation_drawn':False,
              'derived_protocol':'Official test unchanged; lower-priority train/val members excluded by fixed conservative connected components. Original archives/images/official membership retained.',
              'excluded_records':[{'record':by[p],**removed[p]} for p in sorted(removed)],
              'clean_selection_audit_sha256':sha(root/'clean_selection_audit_v1_1.json'),'source_official_manifest_sha256':sha(source),
              'freeze_blockers':['Use the declared fixed conservative grouping and preprocessing protocol.']}
    save(root/'clean_split_proposal_v1_1.json',proposal)
    print(json.dumps({k:v for k,v in report.items() if k not in ('small_classes','class_counts','cross_label_groups')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
