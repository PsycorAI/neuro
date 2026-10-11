#!/bin/bash
# test_cloud_watch.sh -- 2026-10-11 (Neuro 55): local test of cloud/status.sh + cloud/cloud_watch.sh against a fake
# instance tree in /tmp (no ssh, alerts echoed instead of raised).
cd /home/glenn/projects/neuro
F=/tmp/fakecloud; rm -rf $F; mkdir -p $F/logs $F/models/checkpoints/v4_cloud $F/models/checkpoints/v4_cloud_conv
cp logs/v4_smoke1b_fail1.log $F/logs/v4_cloud.log
echo "[run_v4 10-11 00:00:21] 5 failures in a row, giving up" > $F/logs/run_v4.log
export CW_ALERT=echo CW_SEEN=/tmp/fakecloud/seen EVERY=1 INFO_H=1
echo "--- status.sh"; ROOT=$F bash cloud/status.sh | cut -c1-150
echo "--- S1 runner stopped (expect ALERT stopped + INFO, each once)"
CW_CMD="env ROOT=$F bash cloud/status.sh" timeout 4 bash cloud/cloud_watch.sh x 1 | cut -c1-200
echo "--- S2 stall + low tok/s + dense done (fake runner alive, old log)"
touch -d '50 min ago' $F/logs/v4_cloud.log; touch $F/models/checkpoints/v4_cloud/DONE
cp logs/v4_smoke1b_fail1.log $F/logs/v4_cloud_conv.log; touch -d '50 min ago' $F/logs/v4_cloud_conv.log
( exec -a "bash cloud/run_v4.sh FAKE" sleep 8 ) &
sleep 0.5
CW_CMD="env ROOT=$F bash cloud/status.sh" MIN_TPS=9000 timeout 4 bash cloud/cloud_watch.sh x 1 | cut -c1-200
echo "--- S3 all done (expect DONE and exit 0)"
touch $F/models/checkpoints/v4_cloud_conv/DONE
CW_CMD="env ROOT=$F bash cloud/status.sh" timeout 4 bash cloud/cloud_watch.sh x 1 | cut -c1-200; echo "exit=$?"
echo "--- S4 unreachable (expect one ALERT after 3 checks)"
CW_CMD=false timeout 5 bash cloud/cloud_watch.sh x 1 | cut -c1-200
wait; rm -rf $F
