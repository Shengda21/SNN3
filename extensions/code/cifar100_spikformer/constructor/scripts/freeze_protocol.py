"""Verify actual local Git content before experiments; no arbitrary hash or public-registration claim."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
from common import BUNDLE, read, sha, verify_prepared, write


def freeze(root):
    root = verify_prepared(root)
    scientific = list((root/'construction').rglob('identity.json'))
    scientific += [p for p in (root/'training').glob('*/identity.json') if not p.parent.name.startswith('SOFTWARE_')]
    if scientific:
        raise RuntimeError('A prospective freeze cannot be created after experimental completion')
    def git(*args):
        return subprocess.check_output(['git','-C',str(BUNDLE),*args])
    repository = Path(git('rev-parse','--show-toplevel').decode().strip()).resolve()
    commit = git('rev-parse','HEAD').decode().strip()
    required = [BUNDLE/'protocol.json',BUNDLE/'protocols/revision2_prereg.md',
                BUNDLE/'data_manifest.json',BUNDLE/'a_asset_manifest.json',
                BUNDLE/'requirements.lock.txt',BUNDLE/'bootstrap.sh',BUNDLE/'launch.sh']
    allowed = {'.py','.yaml','.yml','.json','.md','.txt','.sh','.csv'}
    required += [p for folder in ('scripts','payload','vendor','upstream') for p in (BUNDLE/folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts
                 and (p.suffix.lower() in allowed or p.name.upper().startswith(('LICENSE','COPYING')))]
    # Large raw datasets and model weights are verified through committed manifests, never committed here.
    data_manifest = read(BUNDLE/'data_manifest.json')
    prepared_split = root/'archive/direction1_budget_rotation_20260907/data/CIFAR100_split.json'
    split_record = next(x for x in data_manifest if x['path']=='CIFAR100_split.json')
    assert sha(prepared_split)==split_record['sha256']
    rows = []
    for path in sorted(required):
        relative = path.resolve().relative_to(repository).as_posix()
        try:
            committed = git('show',f'{commit}:{relative}')
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Commit the complete protocol and code before freezing: {relative}') from exc
        if committed != path.read_bytes():
            raise RuntimeError(f'Working content differs from committed source: {relative}')
        rows.append({'path':path.relative_to(BUNDLE).as_posix(),'sha256':sha(path)})
    record = {'status':'verified_local_git_content_before_experiments','local_git_commit':commit,
        'commit_timestamp':git('show','-s','--format=%cI',commit).decode().strip(),
        'verified_UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(BUNDLE/'protocol.json'),
        'plan_sha256':sha(BUNDLE/'protocols/revision2_prereg.md'),'sources':rows,
        'public_preregistration_claim':False,'meaning':'local prospective content freeze only; publication/remote registration not verified'}
    target = root/'environment/protocol_freeze.json'
    if target.exists():
        previous = read(target)
        assert previous['protocol_sha256']==record['protocol_sha256'] and previous['sources']==rows
        print('Existing content freeze remains verified')
        return
    write(target,record)
    print('Verified local Git content; no remote commit, push or public registration performed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    freeze(parser.parse_args().root)
