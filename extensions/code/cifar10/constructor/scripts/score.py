"""Locked fixed-batch quality replay with saved BN banks and per-example provenance."""
import argparse
import gc
from pathlib import Path
import time
import numpy as np
from common import BUNDLE, environment, read, sha, verify_file, verify_prepared, write


def run(root):
    root = verify_prepared(root)
    environment(root)
    queue = root/'protocol/scoring_queue.json'
    lock = read(root/'protocol/source_lock.json')
    assert lock['queue_sha256'] == sha(queue) and lock['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    assert lock['execution_protocol_sha256'] == sha(root/'environment/protocol.json')
    rows = read(queue)
    import torch
    from torch.nn import functional as F
    from torchvision.datasets import CIFAR10
    from normalization import apply_bank
    import runtime as api
    api.configure()
    ds = CIFAR10(str(api.vision_root(root)/'data'),train=False,transform=api.evaluate_transform())
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor(ds.targets)
    checked = set()
    for row in rows:
        source = (row['checkpoint'],row['checkpoint_sha256'])
        if source not in checked:
            verify_file(*source)
            checked.add(source)
        out = root/'results/scoring'/row['id']
        marker = out/'complete.json'
        if marker.exists():
            done = read(marker)
            assert done['status']=='complete' and done['row']==row and (out/'test.npz').is_file()
            continue
        out.mkdir(parents=True,exist_ok=True)
        tick = time.perf_counter()
        window = row.get('cal_T',row['infer_T'])
        model = api.load_model(root,row['architecture'],row['checkpoint'],window)
        calibration = None
        bank_identity = None
        if row['calibration'] != 'raw':
            bank_path = root/'bn_banks'/f"{row['checkpoint_sha256']}_c{row['cal_T']}_subset9640.pt"
            bank_manifest = root/'bn_banks/manifest.json'
            known_banks = read(bank_manifest) if bank_manifest.exists() else {}
            if bank_path.exists():
                assert bank_path.name in known_banks, 'BN bank exists without a recorded immutable identity'
                verify_file(bank_path,known_banks[bank_path.name]['sha256'])
                saved = torch.load(bank_path,map_location='cpu',weights_only=False)
                assert saved['checkpoint_sha256']==row['checkpoint_sha256'] and saved['cal_T']==row['cal_T'] and saved['cal_seed']==9640
                expected_ids = api.calibration_images(root,9640)[1]
                assert np.array_equal(np.asarray(saved['train_ids']),expected_ids)
                calibration = apply_bank(model,saved['bn_delta'])
            else:
                bank,calibration = api.calibrate(root,model,9640)
                api.atomic_save({'bn_delta':bank,'checkpoint_sha256':row['checkpoint_sha256'],
                    'cal_T':row['cal_T'],'cal_seed':9640,'train_ids':calibration['train_ids']},bank_path)
                known_banks[bank_path.name]={'sha256':sha(bank_path),
                    'checkpoint_sha256':row['checkpoint_sha256'],'cal_T':row['cal_T'],'cal_seed':9640}
                write(bank_manifest,known_banks)
            bank_identity={'file':str(bank_path),'sha256':sha(bank_path)}
            api.set_window(model,row['infer_T'])
            api.prepare_mode(model,False)
        losses,predictions = [],[]
        with torch.no_grad():
            for start in range(0,len(images),4):
                x = images[start:start+4].cuda(non_blocking=True)
                y = labels[start:start+4].cuda(non_blocking=True)
                z = api.forward(model,x)
                assert torch.isfinite(z).all()
                losses.append(F.cross_entropy(z,y,reduction='none').cpu().numpy())
                predictions.append(z.argmax(-1).cpu().numpy())
        loss = np.concatenate(losses).astype(np.float64)
        pred = np.concatenate(predictions).astype(np.int64)
        truth = labels.numpy().astype(np.int64)
        correct = (pred==truth).astype(np.int64)
        assert len(loss)==10000 and np.isfinite(loss).all()
        np.savez_compressed(out/'test.npz',loss_sum=loss,counts=np.ones(10000,dtype=np.int64),
            correct=correct,pred=pred,labels=truth,block_id=np.arange(10000,dtype=np.int64))
        write(marker,{'status':'complete','row':row,'ce':float(loss.mean()),'accuracy':float(correct.mean()),
            'items':10000,'targets':10000,'batch':4,'precision':'FP32','tf32':False,'engine':'eager',
            'calibration':calibration,'bn_bank_identity':bank_identity,'process_seconds':time.perf_counter()-tick,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
        print(row['id'],flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
