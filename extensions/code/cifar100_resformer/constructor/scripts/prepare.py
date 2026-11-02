"""Prepare a clean execution tree from code and the separately supplied local data packet."""
import argparse
from pathlib import Path
import shutil
import tarfile
from common import BUNDLE, protocol, read, sha, verify_prepared, write


def import_A(root, directory):
    root = verify_prepared(root)
    directory = Path(directory).resolve()
    manifest = read(directory/'manifest.json')
    rows = []
    for item in manifest['screen_sources']:
        source = directory/item['path']
        assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']
        target = root/'A_inputs'/item['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert sha(target) == item['sha256']
        else:
            shutil.copy2(source,target)
        rows.append({'seed':item['seed'],'lr':item['lr'],'checkpoint':str(target),
                     'checkpoint_sha256':item['sha256']})
    item = manifest['retained_start']
    source = directory/item['path']
    assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']
    target = root/'A_inputs'/item['path']
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(target) == item['sha256']
    else:
        shutil.copy2(source,target)
    write(root/'protocol/a_mtt_screen_sources.json',rows)
    write(root/'protocol/a_mtt_retained_start.json',{'checkpoint':str(target),'checkpoint_sha256':item['sha256']})
    print('Imported verified historical A inputs; no GPU operation')


def prepare(root, data):
    root, data = Path(root).resolve(), Path(data).resolve()
    if (root/'environment/prepared.json').exists():
        verify_prepared(root)
        print('Already prepared; no training')
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Use an empty execution directory')
    for item in read(BUNDLE/'data_manifest.json'):
        source = data/item['path']
        assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']
    split = read(data/'CIFAR100_split.json')
    a,b = set(split['train']),set(split['validation'])
    assert len(a)==45000 and len(b)==5000 and not a&b and a|b==set(range(50000))
    for name in ('environment','state','logs','protocol','construction','training','models','scoring','analysis','legacy/code'):
        (root/name).mkdir(parents=True,exist_ok=True)
    vision = root/'archive/direction1_budget_rotation_20260907'
    shutil.copytree(BUNDLE/'payload/vision',vision)
    (vision/'data').mkdir(exist_ok=True)
    shutil.copy2(data/'CIFAR100_split.json',vision/'data/CIFAR100_split.json')
    with tarfile.open(data/'cifar-100-python.tar.gz','r:gz') as archive:
        for item in archive.getmembers():
            target = (vision/'data'/item.name).resolve()
            assert target.is_relative_to((vision/'data').resolve())
            if item.isdir():
                target.mkdir(parents=True,exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(item) as source, target.open('wb') as dest:
                    shutil.copyfileobj(source,dest)
            else:
                raise RuntimeError('Archive contains a nonregular entry')
    shutil.copytree(BUNDLE/'payload/legacy_code',root/'legacy/code',dirs_exist_ok=True)
    write(root/'environment/protocol.json',protocol())
    sources = [x for base in ('scripts','vendor','payload','protocols','upstream') for x in (BUNDLE/base).rglob('*') if x.is_file() and '__pycache__' not in x.parts]
    write(root/'environment/prepared.json',{'status':'prepared_not_executed','protocol_sha256':sha(BUNDLE/'protocol.json'),
          'sources':[{'path':x.relative_to(BUNDLE).as_posix(),'sha256':sha(x)} for x in sorted(sources)],
          'runtime_sources':[{'path':x.relative_to(root).as_posix(),'sha256':sha(x)}
              for source_root in (vision,root/'legacy') for x in sorted(source_root.rglob('*'))
              if x.is_file() and 'data' not in x.relative_to(root).parts and '__pycache__' not in x.parts],
          'runtime_data':[{'path':x.relative_to(root).as_posix(),'sha256':sha(x)} for x in sorted((vision/'data').rglob('*')) if x.is_file()],
          'old_results_imported':False,'GPU_experiments_executed':False})
    print('Prepared C/F execution tree; no model construction or training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=BUNDLE.parent/'cloud_revision2_data_20261020')
    parser.add_argument('--A-assets',dest='A_assets',type=Path)
    args = parser.parse_args()
    prepare(args.root,args.data)
    if args.A_assets:
        import_A(args.root,args.A_assets)
