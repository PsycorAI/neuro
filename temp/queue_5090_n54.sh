#!/bin/bash
# queue_5090_n54.sh -- 2026-10-09 (Neuro 54). Robust 5090 chain (replaces queue_5090_r26p.sh). Runs listed in
# temp/queue_5090_n54.list (one run name per line, read again before every run -> append to it to extend the chain
# without touching this script). Per run: wait for the outage queue, train with auto-RESUME (up to 3 attempts after a
# crash / watchdog kill), then the r26n evals, then a verdict line vs r26n. Every outcome goes to temp/alert.sh.
# The name matches the watchdog_tps.sh queue pattern (queue_5090_), so the tok/s watchdog stays alive with it.
# Launch: setsid -f bash temp/queue_5090_n54.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
LIST=temp/queue_5090_n54.list; DONEL=temp/queue_5090_n54.done; touch $DONEL
A(){ bash temp/alert.sh "$@"; }
while pgrep -f "^bash temp/queue_5090_(after_outage|r26p).sh" > /dev/null; do sleep 60; done   # r26p started under the old queue
while :; do
  r=$(grep -v '^#' $LIST | grep -vxFf $DONEL | head -1)
  [ -z "$r" ] && break
  echo "$r" >> $DONEL
  [ -f configs/$r.yaml ] || { A ALERT "$r: no config, skipped"; continue; }
  echo "[n54 $(date '+%m-%d %H:%M:%S')] 5090 chain: $r" >> $LOG
  A INFO "$r started on the 5090"
  for try in 1 2 3; do
    GPU=0 FREE_MIN=1 UNTIL="2026-10-13 06:00" bash temp/run_5080_sched.sh $r
    [ -f models/checkpoints/$r/DONE ] && break
    err=$(grep -E 'Error|Traceback|TRIPPED' logs/$r.log | tail -1 | cut -c1-120)
    if [ $try = 3 ]; then A ALERT "$r FAILED 3 times, chain moves on. Last: $(grep '^step' logs/$r.log | tail -1 | cut -c1-40) $err"
    else A WARN "$r attempt $try ended early ($(grep '^step' logs/$r.log | tail -1 | cut -c1-30) $err) -> auto-resume"; sleep 60; fi
  done
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  grep -qF "[$r] evals + diagnostics DONE" $LOG ||
  { GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  echo "[n54 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG; }
  c1=$(grep "r25-evals $r .*chunk1" $LOG | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
  hay=$(grep "r25-evals $r .*N8:" $LOG | tail -1 | sed -n 's/.*\(N8:.*\)/\1/p' | cut -c1-90)
  val=$(grep 'eval\] val_ppl' logs/$r.log | tail -1 | awk '{print $3}')
  if [ -z "$c1" ]; then A ALERT "$r trained (val $val) but the evals produced no chunk1 line"
  else A DONE "$r chunk1 $c1 (r26n 38.09, $(awk -v a=$c1 'BEGIN{printf "%+.1f%%", (a / 38.09 - 1) * 100}')) val $val | $hay"; fi
done
A DONE "5090 chain finished: the 5090 is idle"
