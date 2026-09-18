"""Archive completed R2.1 evidence and models before the authorized shutdown."""
from pathlib import Path
import argparse,hashlib,json,os,tarfile,time

WORK=Path(os.environ.get('R2_WORK','/root/snn_revision_round2'))
DEST=Path(os.environ.get('R2_BACKUP_FILE','/root/autodl-tmp/revision_round2_complete_20260909.tar'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--records-only',action='store_true');args=ap.parse_args()
    status=json.loads((WORK/'results/finish_status.json').read_text())
    assert status['phase']=='experiments_complete_pending_download',status
    verify=json.loads((WORK/'analysis/verification.json').read_text())
    assert verify['status']=='passed' and verify['scoring_paths']==126 and verify['timing_calls']==2880
    files=sorted(p for p in WORK.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.name not in ['archive_manifest.json','archive_transfer.json'])
    rows=[{'path':p.relative_to(WORK).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    manifest=WORK/'results/archive_manifest.json'
    manifest.write_text(json.dumps({'time':time.time(),'root':str(WORK),'files':rows,
        'bytes':sum(r['bytes'] for r in rows),'scope':'all R2.1 code, protocol, logs, raw scores, timing, analyses and model states'},indent=2))
    temp=DEST.with_suffix('.partial')
    selected=[p for p in files if not args.records_only or p.relative_to(WORK).parts[0]!='models']+[manifest]
    with tarfile.open(temp,'w:gz' if args.records_only else 'w',**({'compresslevel':1} if args.records_only else {})) as tar:
        for p in selected:tar.add(p,arcname='revision_round2/'+p.relative_to(WORK).as_posix(),recursive=False)
    temp.replace(DEST)
    report={'archive':str(DEST),'bytes':DEST.stat().st_size,'sha256':digest(DEST),'files':len(selected),
        'records_only_transfer':args.records_only,'all_source_files_in_manifest':len(rows),'created_at':time.time()}
    (WORK/'results/archive_transfer.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
