"""Prepare a fresh R2.1 execution directory using restored original assets."""
from pathlib import Path
import argparse,json,shutil,tarfile

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination',required=True,type=Path)
    ap.add_argument('--experiment-root',required=True,type=Path,
        help='Prepared original execution tree after extracting the model/data companion')
    args=ap.parse_args();dest=args.destination.resolve();experiment=args.experiment_root.resolve()
    assert not dest.exists(),'Use a new directory so original completion records remain separate.'
    inv=json.loads((ROOT/'revision_round2/protocol/asset_inventory.json').read_text(encoding='utf-8'))
    missing=[r['archive_relative_path'] for r in inv['asset_checks'] if not (experiment/r['archive_relative_path']).is_file()]
    assert not missing,missing
    dest.mkdir(parents=True)
    shutil.copytree(ROOT/'revision_round2/code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'revision_round2/protocol',dest/'protocol',ignore=shutil.ignore_patterns('RUN_LOG.md'))
    for name in ['results','models','logs']:(dest/name).mkdir()
    data=experiment/'direction1_budget_rotation_20260907/data'
    if not (data/'cifar-100-python/train').exists():
        with tarfile.open(data/'cifar-100-python.tar.gz','r:gz') as archive:
            for item in archive.getmembers():
                assert (data/item.name).resolve().is_relative_to(data.resolve()),item.name
            archive.extractall(data,filter='data')
    config={'R2_WORK':str(dest),'R2_ARCHIVE':str(experiment),'state':'prepared; no new model execution',
        'source_protocol':'R2.1','source_code':'unchanged copies of the recorded execution code',
        'hardware_requirement':'two CUDA devices; recorded environment listed in README',
        'instructions':'Set R2_WORK and R2_ARCHIVE before using the execution commands in README.'}
    (dest/'execution_paths.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2))


if __name__=='__main__':main()
