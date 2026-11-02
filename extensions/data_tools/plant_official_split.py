"""Preserve PlantWild v1 official membership; flag contamination without moving samples."""
import argparse,collections,hashlib,json
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def run(root):
    root=Path(root);base=root/'original_extracted';source=base/'plantwild'
    output=root/'official_split_draft_v1.json'
    assert not output.exists(),'Preserve completed split draft'
    classes={}
    for line in (source/'classes.txt').read_text().splitlines():
        if line.strip():
            key,name=line.split(' ',1);classes[int(key)]=name.strip()
    assert set(classes)==set(range(89))
    rows=[json.loads(x) for x in (root/'image_inventory.jsonl').read_text().splitlines()]
    by_path={r['path']:r for r in rows};assert len(by_path)==18542
    official={};mode_name={'0':'test','1':'train','2':'validation'}
    for line in (source/'trainval.txt').read_text().splitlines():
        path,label,mode=line.rsplit('=',2);label=int(label);relative='plantwild/images/'+path
        assert relative in by_path and relative not in official
        assert by_path[relative]['parent_folder']==classes[label]
        official[relative]=(label,mode_name[mode])
    assert set(official)==set(by_path)
    counts=collections.Counter(s for _,s in official.values())
    assert dict(counts)=={'train':13045,'test':3677,'validation':1820}
    urls=json.loads((source/'url_record.json').read_text());assert set(urls)==set(by_path)
    near=json.loads((root/'near_duplicate_audit.json').read_text())
    candidate_group={p:hashlib.sha256(p.encode()).hexdigest() for p in by_path}
    components=near['nontrivial_components']
    crossings=[]
    for component in components:
        key=hashlib.sha256('\n'.join(sorted(component)).encode()).hexdigest()
        for path in component:candidate_group[path]=key
        splits=sorted({official[path][1] for path in component})
        if len(splits)>1:crossings.append({'paths':component,'official_splits':splits,'reason':'exact/perceptual candidate component; not proof of same physical object'})
    url_groups=collections.defaultdict(list)
    for path,url in urls.items():
        if isinstance(url,str) and url.strip():url_groups[url.strip()].append(path)
    repeated_urls=[{'source_page_url':u,'paths':paths,'official_splits':sorted({official[p][1] for p in paths})}
                   for u,paths in url_groups.items() if len(paths)>1]
    url_cross=[g for g in repeated_urls if len(g['official_splits'])>1]
    exact=json.loads((root/'decoded_exact_duplicate_groups.json').read_text())
    exact_cross=[{'decoded_rgb_sha256':key,'paths':paths,'official_splits':sorted({official[p][1] for p in paths})}
                 for key,paths in exact.items() if len({official[p][1] for p in paths})>1]
    report={'status':'complete','GPU_used':False,'official_membership_preserved':True,'no_samples_removed_or_moved':True,
            'cross_official_split_perceptual_components':crossings,'cross_official_split_exact_groups':exact_cross,
            'repeated_source_page_groups':len(repeated_urls),'cross_official_split_source_page_groups':len(url_cross),
            'source_page_warning':'A repeated article URL is provenance, not necessarily the same photo or physical leaf; source domain alone is not a group.',
            'cross_source_pages':url_cross,'model_results_consulted':False,
            'freeze_blockers':['Root must review cross-official-split exact/perceptual candidates before execution freeze.'] if crossings or exact_cross else []}
    save(root/'official_split_contamination_audit.json',report)
    records=[]
    for global_id,path in enumerate(sorted(by_path)):
        label,split=official[path]
        records.append({'global_id':global_id,'path':path,'label':label,'class_name':classes[label],'split':split,
                        'official_split':split,'perceptual_group_id':candidate_group[path],
                        'source_page_url':urls[path],'file_sha256':by_path[path]['file_sha256']})
    proposal={'status':'deterministic_data_reconstruction','dataset':'PlantWild_v1','version':'v1 only',
              'classes':[classes[i] for i in range(89)],'records':records,'actual_counts':dict(counts),
              'class_counts':{s:dict(collections.Counter(r['class_name'] for r in records if r['split']==s)) for s in mode_name.values()},
              'membership_source':'Original plantwild/trainval.txt, interpreted using original readme mode0test/1train/2validation',
              'source_split_sha256':digest(source/'trainval.txt'),'source_classes_sha256':digest(source/'classes.txt'),
              'source_urls_sha256':digest(source/'url_record.json'),'source_inventory_sha256':digest(root/'image_inventory.jsonl'),
              'official_split_preserved':True,'new_validation_drawn':False,'groups_are_not_claimed_split_disjoint':True,
              'freeze_blockers':report['freeze_blockers'],'model_outputs_consulted':False,'test_used_for_tuning':False}
    save(output,proposal)
    print(json.dumps({'status':proposal['status'],'counts':proposal['actual_counts'],'classes':len(classes),
                      'exact_cross_groups':len(exact_cross),'perceptual_cross_components':len(crossings),
                      'source_page_cross_groups':len(url_cross),'split_sha256':digest(output),'GPU_used':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
