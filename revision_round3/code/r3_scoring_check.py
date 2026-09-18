"""Validate the new scoring path using training images only."""
import json,gc
import numpy as np
import torch
from r2_vision import WORK,INIT,configure,write_json
from r2_score import training_inputs
from r3_score import score,calibration,digest

configure();x,y=training_inputs('vision');data=(x,y,torch.arange(len(x)))
cal={9640:calibration(9640)};checks={'vision':x};rows=[]
base={'task':'vision','checkpoint':str(INIT),'checkpoint_sha256':digest(INIT),'split':'training_probe','role':'implementation_check'}
for c,e in [(2,4),(4,2)]:
    for engine in ['eager','graph']:
        row={**base,'id':f'training_probe_c{c}_e{e}_{engine}','infer_T':e,'cal_T':c,
             'cal_seed':9640,'calibration':'bn_cross','engine':engine}
        result=score(row,data,cal,checks)
        assert result['bn_check']['protected_values_and_dtypes_unchanged']
        rows.append({'cal_T':c,'infer_T':e,'engine':engine,'recorded_engine':result['engine']})
        gc.collect();torch.cuda.empty_cache()
    a=np.load(WORK/f'results/scoring/training_probe_c{c}_e{e}_eager/test.npz')
    b=np.load(WORK/f'results/scoring/training_probe_c{c}_e{e}_graph/test.npz')
    assert all(np.array_equal(a[k],b[k]) for k in a.files),(c,e)
write_json(WORK/'results/E0/scoring_check.json',{'status':'passed','split':'training','images':len(x),
    'both_off_diagonal_operations_checked':True,'all_loss_prediction_count_arrays_equal_between_engines':True,
    'protected_parameters_and_non_BN_buffers_verified':True,'rows':rows})
print('TRAINING_SIDE_SCORING_CHECK_PASSED',flush=True)
