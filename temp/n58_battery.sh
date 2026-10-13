#!/bin/bash
# n58_battery.sh [ckpt] -- 2026-10-12 (Neuro 58, Glenn's request). Waits for the 1B checkpoint to be home from Drive,
# then runs the read-only test battery on the 5090: spike diagnostics, memory on / off per chunk, in-window and
# cross-chunk recall, plain-language facts, copy, energy, text samples, self-knowledge probe.
# Each step under a 45 min limit; one DONE alert. Log: logs/n58_battery.log
# Launch: setsid -f bash temp/n58_battery.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
CK=${1:-models/checkpoints/v4_cloud_pin/step_0046500.pt}
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/n58_battery.log
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HAYSTACK_B=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
end=$(( $(date +%s) + 10800 ))
until [ -f "$CK" ] && ! pgrep -f '^rclone copyto' > /dev/null; do
  [ "$(date +%s)" -ge "$end" ] && { bash temp/alert.sh ALERT "n58 battery: $CK not home after 3 h"; exit 1; }
  sleep 30
done
run(){ echo "== $1 ($(date +%H:%M))"; shift; timeout 2700 "$@" 2>&1 | grep -v -i warning | grep -E "SAMPLE|v4|C=|energy|recall|rror|%|N8|N32|rate|per |code|signal|by chunk|probe|Killed" | cut -c1-700; }
{
echo "[battery] $CK start $(date '+%m-%d %H:%M')"
run "spike diagnostics" $PY temp/n58_spike_diag.py --B 4 v4=$CK
run "perplexity per chunk, memory on / off" $PY temp/chunk_eval.py --T 2048 --iters 20 --B 4 v4=$CK
run "haystack in-window (gap 0/64/192/440)" $PY temp/haystack_recall_eval.py v4=$CK
run "cross-chunk haystack" $PY temp/xchunk_haystack_eval.py v4=$CK
run "nl_fact (n 64)" $PY temp/nl_fact_eval.py --n 64 v4=$CK
run "xcopy" $PY temp/xcopy_eval.py v4=$CK
run "energy-v2" $PY temp/energy_eval_v2.py v4=$CK
run "text samples" $PY temp/n57_sample.py $CK
run "self-knowledge probe" $PY scripts/selfknow_probe.py --run v4_cloud_pin
echo "[battery] done $(date '+%m-%d %H:%M')"
} > $LOG 2>&1
bash temp/alert.sh DONE "n58 battery on $(basename $CK) finished: logs/n58_battery.log ($(grep -c -i 'rror\|Killed' $LOG) error lines)"
