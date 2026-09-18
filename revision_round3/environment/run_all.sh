#!/bin/bash
set -e
source /root/autodl-tmp/snn_revision_round3/environment/env.sh
python "$R2_WORK/code/r3_runtime_check.py"
python "$R2_WORK/code/r3_benchmark.py"
CONCURRENCY=$(python -c "import json; print(json.load(open('/root/autodl-tmp/snn_revision_round3/results/E0/throughput.json'))['chosen_concurrency'])")
python "$R2_WORK/code/r3_schedule.py" all --concurrency "$CONCURRENCY"
python "$R2_WORK/code/r3_make_score_queue.py"
python "$R2_WORK/code/r3_score.py" --queue "$R2_WORK/protocol/scoring_queue.json" --part 0 --parts 2 > "$R2_WORK/logs/scoring_0.log" 2>&1 &
SCORE_FIRST=$!
python "$R2_WORK/code/r3_score.py" --queue "$R2_WORK/protocol/scoring_queue.json" --part 1 --parts 2 > "$R2_WORK/logs/scoring_1.log" 2>&1 &
SCORE_SECOND=$!
wait "$SCORE_FIRST"
wait "$SCORE_SECOND"
python "$R2_WORK/code/r3_analyze.py"
python "$R2_WORK/code/r3_finalize.py"
python -c "from pathlib import Path; import time,json; Path('/root/autodl-tmp/snn_revision_round3/results/GPU_COMPLETE.json').write_text(json.dumps({'completed_at':time.time(),'status':'complete'}))"
