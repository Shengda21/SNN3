"""New-round identities, hashes and CPU-only configuration helpers."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def protocol():
    p = read(BUNDLE/'protocol.json')
    assert p['budgets'] == [512,1024,2048] and p['updates'] == 2048
    assert len(p['native_rates']) == 7 and len(p['mtt_rates']) == 5
    assert set(p['cohorts']) == {'C-conf','F-arch'}
    for cohort, c in p['cohorts'].items():
        assert len(c['screen_seeds']) == 2 and len(c['formal_seeds']) == 3
        assert set(c['screen_seeds']).isdisjoint(c['formal_seeds'])
    return p


def environment(root):
    root = Path(root).resolve()
    os.environ.update(R2_ARCHIVE=str(root/'archive'), R2_WORK=str(root/'legacy'),
                      SNN_RUN_ROOT=str(root), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    return root


def verify_prepared(root):
    root = Path(root).resolve()
    record = read(root/'environment/prepared.json')
    assert record['protocol_sha256'] == sha(BUNDLE/'protocol.json'), 'Protocol changed: use a fresh root'
    for row in record['sources']:
        assert sha(BUNDLE/row['path']) == row['sha256'], f"Source changed: {row['path']}"
    for row in record['runtime_sources']:
        assert sha(root/row['path']) == row['sha256'], f"Copied runtime source changed: {row['path']}"
    for row in record['runtime_data']:
        assert sha(root/row['path']) == row['sha256'], f"Prepared data changed: {row['path']}"
    assert read(root/'environment/protocol.json') == protocol()
    return root


def verify_file(path, fingerprint):
    assert Path(path).is_file() and sha(path) == fingerprint, f'Changed/missing source: {path}'


def require_frozen(root):
    frozen = read(Path(root)/'environment/protocol_freeze.json')
    assert frozen['status'] == 'verified_local_git_content_before_experiments'
    assert frozen['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    assert frozen['plan_sha256'] == sha(BUNDLE/'protocols/revision2_prereg.md')
    for row in frozen['sources']:
        assert sha(BUNDLE/row['path']) == row['sha256'], 'Code changed after freeze'
    return frozen


@contextmanager
def exclusive(root):
    import fcntl
    path = Path(root)/'state/execution.lock'
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)
