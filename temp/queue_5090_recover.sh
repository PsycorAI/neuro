#!/bin/bash
# queue_5090_recover.sh -- 2026-10-10 21:30 (Neuro 55; power loss at ~19:00, Glenn: resume the 5090 runs).
# r26u was at step 15000 of 17089 (ckpt step_0015000.pt). queue_spk_conv.sh cannot be reused (it re-seeds: rm -rf),
# so this RESUMES r26u (train_gpu auto-resume, log appended), runs the same evals + verdict as queue_spk_conv.sh /
# queue_5090_smoke1b.sh step 2, then hands over to queue_5090_smoke1b_rerun.sh (1B smoke rerun -> n54 chain:
# r26v, r26w, r26x). Launch: setsid -f bash temp/queue_5090_recover.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
export T=2048; PYB=/home/glenn/projects/bdh/venv/bin/python; PY=$PYB; LOG=logs/gap_queue.log; A(){ bash temp/alert.sh "$@"; }
G=0; r=r26u_13L_RC3_conv_llamakd_s1_150m; SRC=$(ls models/checkpoints/r26r_13L_RC3_seed1_150m/step_*.pt | tail -1)
ls models/checkpoints/$r/step_*.pt > /dev/null 2>&1 && [ -n "$SRC" ] || { A ALERT "$r: no checkpoint to resume from"; exit 1; }
echo "[n55 $(date '+%m-%d %H:%M:%S')] [$r] RESUME after the power loss ($(ls models/checkpoints/$r/step_*.pt | tail -1))" >> $LOG
A INFO "$r resumed on the 5090 after the power loss"
touch logs/$r.log        # the tok/s watchdog reads a stale log mtime as a hang (21:29: "silent 151 min" -> SIGTERM)
CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 $PY src/train_gpu.py --config configs/$r.yaml --device cuda >> logs/$r.log 2>&1
echo "[g$G $(date '+%m-%d %H:%M:%S')] [$r] exit=$? :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=$G $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 | sed "s/^/  recall /" >> $LOG
CUDA_VISIBLE_DEVICES=$G $PY temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=$G bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
CUDA_VISIBLE_DEVICES=$G $PY temp/energy_eval.py "$r=$ck" "src=$SRC" 2>&1 | grep energy | sed "s/^/  energy /" >> $LOG
echo "[g$G $(date '+%m-%d %H:%M:%S')] $r queue DONE" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PYB temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
c1=$(grep "r25-evals $r .*chunk1" $LOG | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
hay=$(grep "r25-evals $r .*N8:" $LOG | tail -1 | sed -n 's/.*\(N8:.*\)/\1/p' | cut -c1-80)
en=$(grep "$r  *energy-v2" $LOG | tail -1 | sed -n 's/.*compute only \([0-9.]*x\).*SRAM fetch \([0-9.]*x\).*/\1 \/ \2 SRAM/p')
if [ -n "$c1" ]; then A DONE "$r chunk1 $c1 (r26o seed 0: 36.55; its dense r26r: 38.65; RESUMED once at step 15000) energy ${en:-?} | $hay"
else A ALERT "$r produced no chunk1 line: $(grep -E 'Error|Traceback' logs/$r.log | tail -1 | cut -c1-120)"; fi
exec bash temp/queue_5090_smoke1b_rerun.sh
