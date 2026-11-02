"""Outcome-independent CPU duplicate audit and an fixed grouped EuroSAT split reconstruction."""
import argparse, collections, hashlib, json, os
from pathlib import Path
import numpy as np

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def run(root):
    root=Path(root); assert not (root/'near_duplicate_audit.json').exists(), 'Preserve completed audit'; rows=[json.loads(x) for x in (root/'image_inventory.jsonl').read_text().splitlines()]
    n=len(rows);parents=list(range(n))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j:parents[max(i,j)]=min(i,j)
    exact={}
    for i,r in enumerate(rows):
        key=r['decoded_rgb_sha256']
        if key in exact:union(i,exact[key])
        else:exact[key]=i
    dh=np.array([int(r['dhash64'],16) for r in rows],dtype=np.uint64)
    ph=np.array([int(r['phash64'],16) for r in rows],dtype=np.uint64)
    pair_count=0;cross_class_pairs=0;stored=[]
    for i in range(n-1):
        candidates=np.flatnonzero(np.bitwise_count(dh[i+1:]^dh[i])<=4)+i+1
        candidates=candidates[np.bitwise_count(ph[candidates]^ph[i])<=4]
        for j in candidates.tolist():
            union(i,j);pair_count+=1
            cross_class_pairs+=rows[i]['parent_folder']!=rows[j]['parent_folder']
            if len(stored)<50000:stored.append({'a':rows[i]['path'],'b':rows[j]['path'],'dhash_distance':int((int(dh[i])^int(dh[j])).bit_count()),'phash_distance':int((int(ph[i])^int(ph[j])).bit_count()),'cross_class':rows[i]['parent_folder']!=rows[j]['parent_folder']})
    groups=collections.defaultdict(list)
    for i in range(n):groups[find(i)].append(i)
    components=[v for v in groups.values()]
    report={'status':'complete','GPU_used':False,'pid':os.getpid(),'samples':n,'near_candidate_pairs':pair_count,
            'cross_class_candidate_pairs':cross_class_pairs,'stored_pair_limit':50000,'stored_pairs_truncated':pair_count>50000,
            'rule':'decoded-RGB exact identity OR (dHash64 Hamming<=4 AND pHash64 Hamming<=4); transitive connected components',
            'not_a_claim':'Perceptual candidates are conservative grouping, not proof of same photographed object or geographic independence',
            'nontrivial_components':[[rows[i]['path'] for i in g] for g in components if len(g)>1],
            'largest_component':max(map(len,components)),'pairs':stored}
    save(root/'near_duplicate_audit.json',report)
    if root.name!='eurosat_rgb':return
    classes=sorted({r['parent_folder'] for r in rows});class_index={c:i for i,c in enumerate(classes)}
    total=collections.Counter(r['parent_folder'] for r in rows)
    ratios={'train':.7,'validation':.15,'test':.15};counts={s:collections.Counter() for s in ratios}
    def group_id(g):return hashlib.sha256('\n'.join(rows[i]['path'] for i in g).encode()).hexdigest()
    components.sort(key=lambda g:(-len(g),hashlib.sha256(('20261030:'+group_id(g)).encode()).hexdigest()))
    assigned={};component_ids={}
    for g in components:
        vector=collections.Counter(rows[i]['parent_folder'] for i in g)
        score=lambda s:sum(k*(ratios[s]*total[c]-counts[s][c])/(ratios[s]*total[c]) for c,k in vector.items())
        split=max(ratios,key=score)
        counts[split].update(vector)
        for i in g:assigned[i]=split;component_ids[i]=group_id(g)
    manifest=[{'global_id':i,'path':r['path'],'label':class_index[r['parent_folder']],'class_name':r['parent_folder'],
               'split':assigned[i],'group_id':component_ids[i],'file_sha256':r['file_sha256']} for i,r in enumerate(rows)]
    assert len({r['global_id'] for r in manifest})==n
    assert all(len({assigned[i] for i in g})==1 for g in components)
    proposal={'status':'deterministic_data_reconstruction','dataset':'EuroSAT_RGB','split_seed':20261030,
              'ratios_requested':ratios,'actual_counts':{s:sum(c.values()) for s,c in counts.items()},
              'class_counts':{s:dict(c) for s,c in counts.items()},'classes':classes,'records':manifest,
              'grouping_rule':report['rule'],'source_inventory_sha256':hashlib.sha256((root/'image_inventory.jsonl').read_bytes()).hexdigest(),
              'geography':'RGB archive contains no coordinate sidecars and no JPEG GPS; this is grouped class-stratified in-distribution sampling, not geographic OOD evaluation',
              'near_pairs_are_not_deleted':True,'model_outputs_consulted':False,'test_used_for_tuning':False}
    save(root/'proposed_split_v1.json',proposal)
    print(json.dumps({k:v for k,v in proposal.items() if k not in ('records','class_counts')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
