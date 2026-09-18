"""Audit saved R3 scoring provenance and outputs without executing a GPU model."""
from pathlib import Path
from collections import Counter
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'revision_round3'


def main():
    queue = json.loads((WORK / 'protocol/scoring_queue.json').read_text())
    lock = json.loads((WORK / 'results/model_analysis_lock.json').read_text())
    engines, cohorts = Counter(), Counter()
    graph_checks = bn_checks = 0
    max_ce_error = max_accuracy_error = 0.0
    identities = set()
    for row in queue:
        folder = WORK / 'results/scoring' / row['id']
        meta = json.loads((folder / 'complete.json').read_text())
        assert meta['row'] == row, row['id']
        source = row['checkpoint'] or row['identity_path']
        assert lock['source_identities'][source] == row['checkpoint_sha256']
        identities.add(row['checkpoint_sha256'])
        assert meta['status'] == 'complete'
        with np.load(folder / 'test.npz') as a:
            assert np.isfinite(a['loss_sum']).all()
            targets = int(a['counts'].sum())
            correct = int(a['correct'].sum())
            assert targets == (40980 if row['task'] == 'language' else 10000)
            assert int((a['pred'] == a['labels']).sum()) == correct
            ce_error = abs(float(a['loss_sum'].sum() / targets) - meta['ce'])
            accuracy_error = abs(correct / targets - meta['accuracy'])
            assert ce_error < 1e-12 and accuracy_error < 1e-12
            max_ce_error = max(max_ce_error, ce_error)
            max_accuracy_error = max(max_accuracy_error, accuracy_error)
        engines[meta['engine']] += 1
        cohorts['legacy_replay' if row['role'] == 'legacy_replay' else row['init_id']] += 1
        if meta['engine'] == 'graph':
            checks = meta['graph_check']['ABA']
            assert len(checks) == 3 and all(c['bitwise_equal'] for c in checks)
            graph_checks += 1
        if 'bn_check' in meta:
            assert meta['bn_check']['protected_values_and_dtypes_unchanged']
            bn_checks += 1
    assert dict(cohorts) == {'A': 234, 'B': 146, 'legacy_replay': 126}
    assert len(identities) == 83 and len(queue) == 506
    report = {
        'status': 'passed', 'scoring_paths': len(queue),
        'cohorts': dict(cohorts), 'source_checkpoint_identities': len(identities),
        'queue_rows_equal_saved_metadata': True, 'sources_match_model_lock': True,
        'saved_arrays_match_recorded_ce_and_accuracy': True,
        'maximum_absolute_ce_error': max_ce_error,
        'maximum_absolute_accuracy_error': max_accuracy_error,
        'engines': dict(engines), 'training_ABA_bitwise_graph_checks': graph_checks,
        'BN_protected_state_checks': bn_checks,
        'scope': 'Local validation of recorded outputs and archived execution checks. '
                 'This does not perform a second full-test GPU replay.'
    }
    out = ROOT / 'recomputed/revision_round3/record_verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
