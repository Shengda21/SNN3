"""Stage only already-local CIFAR inputs or the eleven needed A-screen/start weights."""
import argparse
from pathlib import Path
import shutil
from common import BUNDLE, read, sha, write


def build_data(old, destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Choose an empty asset destination')
    destination.mkdir(parents=True,exist_ok=True)
    rows = []
    for name in ('cifar-100-python.tar.gz','CIFAR100_split.json'):
        source = old/'assets/raw/vision'/name
        target = destination/name
        shutil.copy2(source,target)
        rows.append({'path':name,'bytes':target.stat().st_size,'sha256':sha(target)})
    write(destination/'manifest.json',{'kind':'existing_CIFAR100_raw_and_fixed_split','files':rows})
    write(BUNDLE/'data_manifest.json',rows)
    print(f'Staged {len(rows)} existing data inputs; no network')


def build_A(source, destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Choose an empty A-asset destination')
    destination.mkdir(parents=True,exist_ok=True)
    rows = []
    rates = (5e-6,2e-5,8e-5,2e-4,5e-4)
    for seed in (9611,9612):
        for lr in rates:
            name = f'mtt_s{seed}_lr{lr:g}'
            old = source/'r3/models/mtt_screen'/name/'step2048.pt'
            target = destination/'mtt_screen'/name/'step2048.pt'
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old,target)
            rows.append({'seed':seed,'lr':lr,'path':target.relative_to(destination).as_posix(),
                         'bytes':target.stat().st_size,'sha256':sha(target)})
    old = source/'archive/direction1_budget_rotation_20260907/results/reconstructed_identity_p0.0/best.pt'
    target = destination/'starts/S0_A.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(old,target)
    start = {'path':target.relative_to(destination).as_posix(),'bytes':target.stat().st_size,'sha256':sha(target)}
    manifest = {'kind':'A_existing_raw_selected_MTT_screens_and_retained_start',
                'screen_sources':rows,'retained_start':start,'old_raw_selected_rate':8e-5,
                'scientific_role':'inputs for a new calibrated validation selection; not new results'}
    write(destination/'manifest.json',manifest)
    write(BUNDLE/'a_asset_manifest.json',manifest)
    print('Staged 10 existing MTT screening endpoints and S0_A; no training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['data','A'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--destination',type=Path)
    a = parser.parse_args()
    if a.kind == 'data':
        build_data((a.source or BUNDLE.parent/'server_rerun_single_gpu_20261016').resolve(),
                   (a.destination or BUNDLE.parent/'cloud_revision2_data_20261020').resolve())
    else:
        build_A((a.source or BUNDLE.parent/'unified_results_20261016').resolve(),
                (a.destination or BUNDLE.parent/'cloud_revision2_A_assets_20261020').resolve())
