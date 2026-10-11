#!/bin/bash
# queue_5090_smoke1b.sh -- 2026-10-10 (Neuro 54; Glenn: smoke RC3 at the 1B shape on the 5090 before the H200).
# After the n54 chain (r26t -> r26s) has exited:
#  1. 1B SMOKE: the real cloud/run_v4.sh on configs/v4_smoke1b.yaml -> v4_smoke1b_conv.yaml (150 + 60 steps), with one
#     planned stop + resume after step 50 (temp/kill_at_step.sh).
#  2. r26u = RC3 conversion from r26r (seed 1): second seed of the 36.55 headline (temp/queue_spk_conv.sh + energy-v2).
# The name matches the watchdog_tps.sh queue pattern (queue_5090_). Reports through temp/alert.sh.
# Launch: setsid -f bash temp/queue_5090_smoke1b.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
export T=2048; PYB=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log; A(){ bash temp/alert.sh "$@"; }
while pgrep -f "^bash temp/queue_5090_n54.sh" > /dev/null; do sleep 60; done
# ---- 1. 1B smoke ----
rm -rf models/checkpoints/v4_smoke1b models/checkpoints/v4_smoke1b_conv; : > logs/v4_smoke1b.log; : > logs/v4_smoke1b_conv.log
echo "[n54 $(date '+%m-%d %H:%M:%S')] 5090: 1B smoke v4_smoke1b -> v4_smoke1b_conv (cloud/run_v4.sh)" >> $LOG
A INFO "1B smoke started on the 5090"
setsid -f bash temp/kill_at_step.sh v4_smoke1b 50 < /dev/null > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 ROOT=/home/glenn/projects/neuro PY=$PYB HF_HOME=/home/glenn/.cache/huggingface \
  bash cloud/run_v4.sh v4_smoke1b v4_smoke1b_conv
par=$(grep -o 'params=[0-9.]*M' logs/v4_smoke1b.log | tail -1)
tps=$(grep '^step' logs/v4_smoke1b.log | tail -3 | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr '\n' ' ')
vr=$(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/v4_smoke1b.log | tail -1); vrc=$(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/v4_smoke1b_conv.log | tail -1)
vd=$(grep 'eval\] val_ppl' logs/v4_smoke1b.log | awk '{print $3}' | tr '\n' ' ')
vc=$(grep 'eval\] val_ppl' logs/v4_smoke1b_conv.log | awk '{print $3}' | tr '\n' ' ')
res=$(grep -c '^resumed' logs/v4_smoke1b.log)
if [ -f models/checkpoints/v4_smoke1b_conv/DONE ]; then
  A DONE "1B SMOKE PASSED ($par): dense val $vd($vr, $tps tok/s, $res resume) -> seed -> conversion val $vc($vrc)"
else
  A ALERT "1B SMOKE FAILED ($par, dense val ${vd:-none}): $(tail -n 1 logs/run_v4.log | cut -c1-100) | $(grep -hE 'Error|Traceback' logs/v4_smoke1b.log logs/v4_smoke1b_conv.log | tail -1 | cut -c1-120)"
fi
rm -rf models/checkpoints/v4_smoke1b models/checkpoints/v4_smoke1b_conv        # 2 x ~13 GB smoke checkpoints
# ---- 2. conversion, seed 1 ----
r=r26u_13L_RC3_conv_llamakd_s1_150m; SRC=$(ls models/checkpoints/r26r_13L_RC3_seed1_150m/step_*.pt | tail -1)
[ -f configs/$r.yaml ] && [ -n "$SRC" ] || { A ALERT "$r: config or r26r checkpoint missing"; exit 1; }
A INFO "$r started on the 5090"
bash temp/queue_spk_conv.sh 0 $r $SRC
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PYB temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
c1=$(grep "r25-evals $r .*chunk1" $LOG | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
hay=$(grep "r25-evals $r .*N8:" $LOG | tail -1 | sed -n 's/.*\(N8:.*\)/\1/p' | cut -c1-80)
en=$(grep "$r  *energy-v2" $LOG | tail -1 | sed -n 's/.*compute only \([0-9.]*x\).*SRAM fetch \([0-9.]*x\).*/\1 \/ \2 SRAM/p')
if [ -n "$c1" ]; then A DONE "$r chunk1 $c1 (r26o seed 0: 36.55; its dense r26r: 38.65) energy ${en:-?} | $hay"
else A ALERT "$r produced no chunk1 line: $(grep -E 'Error|Traceback' logs/$r.log | tail -1 | cut -c1-120)"; fi
A DONE "5090 smoke + conversion queue finished: the 5090 is idle"
