"""Validate the finite queue and prepare the complete experiment download inventory."""
import hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

WORK=Path(os.environ['R2_WORK'])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    report=json.loads((WORK/'analysis/summary.json').read_text());assert report['status']=='passed'
    for name in ['runtime_check.json','scoring_check.json','asset_verification.json','source_verification.json']:
        assert json.loads((WORK/'results/E0'/name).read_text())['status']=='passed'
    language=json.loads((WORK/'results/E0/language_migration.json').read_text())
    assert {row['T'] for row in language}=={2,4}
    assert all(check['bitwise_equal'] for row in language for check in row['ABA'])
    for name in ['screen_queue.json','formal_queue.json','second_queue.json']:
        queue=json.loads((WORK/'protocol'/name).read_text())
        for job in queue:
            if job.get('kind')=='initial':record=WORK/'initialization/results/independent12550/complete.json'
            else:record=WORK/'results'/job['stage']/job['tag']/'complete.json'
            done=json.loads(record.read_text());assert done['status']=='complete'
            if job.get('kind')!='initial':
                assert done['step']==2048
                for K in [512,1024,2048]:assert (WORK/'models'/job['stage']/job['tag']/f'step{K}.pt').is_file()
    rows=json.loads((WORK/'protocol/scoring_queue.json').read_text())
    assert len(rows)==report['scoring_paths']
    for row in rows:
        meta=json.loads((WORK/'results/scoring'/row['id']/'complete.json').read_text())
        assert meta['status']=='complete'
        if row['calibration']!='raw':assert meta['bn_check']['protected_values_and_dtypes_unchanged']
        if meta['engine']=='graph':assert all(q['bitwise_equal'] for q in meta['graph_check']['ABA'])
    files=[]
    for directory in ['code','protocol','results','models','analysis','initialization/results','environment','logs']:
        root=WORK/directory
        for f in root.rglob('*'):
            rel=f.relative_to(WORK)
            if not f.is_file() or f.is_symlink():continue
            if any(x in {'__pycache__','tmp','wheels','cupy_cache','pipeline_started.lock'} for x in rel.parts):continue
            if f.suffix in {'.tmp','.partial','.pyc'}:continue
            if f.name in {'download_inventory.json','GPU_COMPLETE.json','queue_status.json','scoring_status.json','pipeline.log'}:continue
            files.append(f)
    files=sorted(set(files))
    def record(f):return {'path':f.relative_to(WORK).as_posix(),'bytes':f.stat().st_size,'sha256':digest(f)}
    with ThreadPoolExecutor(max_workers=6) as pool:entries=list(pool.map(record,files))
    inventory={'status':'ready_for_download','created_at':time.time(),'files':entries,
               'total_bytes':sum(x['bytes'] for x in entries),'validated_scoring_paths':len(rows),
               'external_inputs':'Previously verified original/R2.1 model/data assets; not duplicated in new output inventory.'}
    path=WORK/'results/download_inventory.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(inventory,indent=2));temp.replace(path)
    print(json.dumps({'files':len(entries),'bytes':inventory['total_bytes'],'status':'ready_for_download'}),flush=True)
if __name__=='__main__':main()
