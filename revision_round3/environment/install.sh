#!/bin/bash
set -e
export PATH=/root/autodl-tmp/r3_env/bin:/root/miniconda3/bin:/usr/bin:/bin
export TMPDIR=/root/autodl-tmp/snn_revision_round3/environment/tmp
mkdir -p "$TMPDIR"
python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir 'cupy-cuda12x==13.6.0'
python -m pip freeze > /root/autodl-tmp/snn_revision_round3/environment/pip_freeze.txt
printf 'INSTALL_COMPLETE\n'
