"""Replay the locked R2.1 states on local CUDA hardware in a new result directory."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parent
OLD_WORK='/root/snn_revision_round2/'
OLD_ARCHIVE='/root/autodl-tmp/snn0907/'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def relocate(value):
    if value is None:return None
    if value.startswith(OLD_WORK):return str(ROOT/'revision_round2'/value.removeprefix(OLD_WORK))
    if value.startswith(OLD_ARCHIVE):return str(ROOT/'experiment'/value.removeprefix(OLD_ARCHIVE))
    raise ValueError('Unexpected recorded asset path: '+value)


def check_code(directory):
    manifest=read(directory/'results/model_analysis_lock.json')
    for name,expected in manifest['execution_and_analysis_code_sha256'].items():
        assert hashlib.sha256((directory/'code'/name).read_bytes()).hexdigest()==expected,name
    for row in read(directory/'protocol/source_identities.json'):
        assert hashlib.sha256((ROOT/'experiment'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']


def prepare(destination):
    assert not destination.exists(),'Choose a new directory; recorded results must remain intact.'
    subprocess.run([sys.executable,str(ROOT/'verify_revision_assets.py'),'--folder',str(ROOT)],check=True)
    shutil.copytree(ROOT/'revision_round2/code',destination/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'revision_round2/protocol',destination/'protocol',ignore=shutil.ignore_patterns('RUN_LOG.md'))
    for name in ['results','logs','provenance']:(destination/name).mkdir()
    original=ROOT/'revision_round2/results/model_analysis_lock.json'
    shutil.copy2(original,destination/'provenance/model_analysis_lock.original.json')
    manifest=read(original)
    for row in manifest['scoring']:
        row['checkpoint']=relocate(row['checkpoint'])
        row['identity_path']=relocate(row['identity_path'])
    for row in manifest['timing_sessions']:
        row['checkpoints']={T:relocate(path) for T,path in row['checkpoints'].items()}
    manifest['state_identities']={relocate(path):sha for path,sha in manifest['state_identities'].items()}
    manifest['replay_provenance']={'prepared_at_unix':time.time(),
        'original_manifest_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'change':'Asset paths relocated; states, scoring rows, contrasts, budgets and timing sessions unchanged.',
        'data_role':'Re-evaluation of the published fixed inputs; no new model selection or confirmation claim.'}
    write(destination/'results/model_analysis_lock.json',manifest)
    write(destination/'execution_paths.json',{'R2_WORK':str(destination),'R2_ARCHIVE':str(ROOT/'experiment'),
        'scope':'Recorded states, fresh scores and timing; no training',
        'source_package':str(ROOT),'state':'prepared; model execution has not started'})
    check_code(destination)
    # The retained compressed CIFAR asset is expanded only when a GPU replay is prepared.
    data=ROOT/'experiment/direction1_budget_rotation_20260907/data'
    if not (data/'cifar-100-python/train').exists():
        import tarfile
        with tarfile.open(data/'cifar-100-python.tar.gz','r:gz') as archive:
            for member in archive.getmembers():assert (data/member.name).resolve().is_relative_to(data.resolve())
            archive.extractall(data,filter='data')
    print('Prepared recorded-state replay: '+str(destination))


def run(directory):
    paths=read(directory/'execution_paths.json')
    assert paths['source_package']==str(ROOT),'Run from the package used during preparation.'
    check_code(directory)
    env=os.environ.copy();env.update(R2_WORK=str(directory),R2_ARCHIVE=paths['R2_ARCHIVE'],
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    manifest=read(directory/'results/model_analysis_lock.json')
    def command(name,args,gpu=None):
        child=env.copy()
        if gpu is not None:child['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (directory/'logs'/f'{name}.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable]+args,env=child,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see {directory / "logs" / (name+".log")}')
    def status(phase,**extra):
        write(directory/'results/replay_status.json',{'phase':phase,'time':time.time(),**extra})
    try:
        status('training_side_implementation_check')
        command('language_checks',[str(directory/'code/r2_score.py'),'check'],0)
        status('full_scoring')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(command,f'scoring_gpu{gpu}',
                [str(directory/'code/r2_score.py'),'worker','--gpu',str(gpu)],gpu) for gpu in [0,1]]
            for future in futures:future.result()
        for row in manifest['scoring']:
            assert (directory/'results/scoring'/row['id']/'complete.json').is_file(),row['id']
        write(directory/'results/scoring_finished.json',{'paths':len(manifest['scoring']),'time':time.time()})
        status('isolated_timing')
        for session in manifest['timing_sessions']:
            command('timing_'+session['id'],[str(directory/'code/r2_timing.py'),'--id',session['id']],session['gpu'])
        status('analysis')
        command('analysis',[str(directory/'code/r2_analyze.py'),'--root',str(directory)])
        status('complete',scope='New scores and process timings for the recorded model states')
    except Exception:
        status('failed',error=traceback.format_exc());raise
    print('Completed recorded-state replay: '+str(directory))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare').add_argument('--destination',required=True,type=Path)
    sub.add_parser('run').add_argument('--directory',required=True,type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.destination.resolve())
    else:run(args.directory.resolve())
