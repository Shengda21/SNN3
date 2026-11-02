"""Matched-BN validation selection, never test-set scoring."""
import argparse
import gc
from pathlib import Path
from common import BUNDLE, environment, read, sha, verify_file, verify_prepared, write


def evaluate(root,architecture,checkpoint,output):
    root = verify_prepared(root)
    environment(root)
    source = Path(checkpoint).resolve()
    fingerprint = sha(source)
    identity = {'checkpoint_sha256':fingerprint,'architecture':architecture,
                'protocol_sha256':sha(BUNDLE/'protocol.json'),
                'runtime_sha256':sha(BUNDLE/'scripts/runtime.py'),
                'criterion':'mean matched-BN validation CE over e=2,4','calibration_seed':9640}
    output = Path(output)
    if output.exists():
        cached = read(output)
        assert cached['identity'] == identity
        return cached
    import torch
    import runtime as api
    api.configure()
    values = {}
    for window in (2,4):
        model = api.load_model(root,architecture,source,window)
        bank,report = api.calibrate(root,model,9640)
        value = api.evaluate_validation(root,model)
        assert value['split'] == 'validation' and value['samples'] == 5000
        assert report['samples'] == 2048 and report['protected_values_and_dtypes_unchanged']
        values[str(window)] = {'validation':value,'calibration':report}
        api.atomic_save({'bn_delta':bank,'checkpoint_sha256':fingerprint,'cal_T':window,
                        'cal_seed':9640,'train_ids':report['train_ids']},
                       root/'selection_banks'/f'{fingerprint}_c{window}_subset9640.pt')
        del model,bank
        gc.collect()
        torch.cuda.empty_cache()
    result = {'identity':identity,'checkpoint':str(source),'windows':values,
              'joint_matched_BN_validation_ce':sum(v['validation']['ce'] for v in values.values())/2,
              'test_used':False}
    write(output,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--architecture',required=True,choices=['resformer','spikformer'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    evaluate(a.root,a.architecture,a.checkpoint,a.output)
