#!/bin/bash
set -e
while ! grep -q INSTALL_COMPLETE /root/autodl-tmp/snn_revision_round3/logs/install.log; do
    if ! kill -0 1703 2>/dev/null; then echo INSTALL_FAILED; exit 1; fi
    sleep 5
done
mkdir /root/autodl-tmp/snn_revision_round3/environment/pipeline_started.lock
exec /bin/bash /root/autodl-tmp/snn_revision_round3/environment/run_all.sh
