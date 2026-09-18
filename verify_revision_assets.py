"""Verify the R2.1 model/data companion against the recorded experiment identities."""
from pathlib import Path
import argparse,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive',type=Path,help='model_data_assets.zip; verify without extracting')
    group.add_argument('--folder',type=Path,help='Extracted companion root containing experiment/ and revision_round2/')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    inv=read(ROOT/'revision_round2/protocol/asset_inventory.json')
    lock=read(ROOT/'revision_round2/results/model_analysis_lock.json')
    expected={'experiment/'+r['archive_relative_path']:(r['recorded_sha256'],r['recorded_size']) for r in inv['asset_checks']}
    for remote,sha in lock['state_identities'].items():
        if remote.startswith('/root/snn_revision_round2/'):
            relative='revision_round2/'+remote.removeprefix('/root/snn_revision_round2/')
            expected[relative]=(sha,None)
    assert len(expected)==42
    rows=[]
    archive=zipfile.ZipFile(args.archive) if args.archive else None
    try:
        for name,(sha,size) in expected.items():
            exists=name in archive.namelist() if archive else (args.folder/name).is_file()
            if not exists:
                rows.append({'path':name,'status':'missing'});continue
            with archive.open(name) if archive else (args.folder/name).open('rb') as f:
                h=hashlib.sha256();count=0
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    h.update(block);count+=len(block)
            match=h.hexdigest()==sha and (size is None or size==count)
            rows.append({'path':name,'status':'match' if match else 'mismatch','bytes':count})
    finally:
        if archive:archive.close()
    report={'status':'passed' if all(r['status']=='match' for r in rows) else 'failed',
        'required_files':42,'matches':sum(r['status']=='match' for r in rows),'files':rows,
        'scope':'actual model and data bytes; no GPU execution'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
