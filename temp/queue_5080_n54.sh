#!/bin/bash
# queue_5080_n54.sh -- 2026-10-10 (Neuro 54). 5080 twin of queue_5090_n54.sh: runs from temp/queue_5080_n54.list (read
# again before every run), auto-RESUME up to 3 attempts, r25 evals, result through temp/alert.sh. Stops for Glenn:
# UNTIL below (run_5080_sched.sh: no start within 25 min of it, SIGINT pause 5 min before, then this queue exits).
# 7L runs only (12L spills the 16 GB card).
# Launch: UNTIL="2026-10-10 17:30" setsid -f bash temp/queue_5080_n54.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256; LOG=logs/gap_queue.log
U=${UNTIL:?set UNTIL}; END=$(date -d "$U" +%s)
LIST=temp/queue_5080_n54.list; DONEL=temp/queue_5080_n54.done; touch $DONEL
A(){ bash temp/alert.sh "$@"; }
while pgrep -f "^bash temp/queue_5080_reh.sh" > /dev/null; do sleep 60; done
while :; do
  r=$(grep -v '^#' $LIST | grep -vxFf $DONEL | head -1)
  [ -z "$r" ] && break
  [ $(date +%s) -ge $((END - 1500)) ] && { A INFO "5080 chain: window ends $U, $r not started"; exit 0; }
  echo "$r" >> $DONEL
  [ -f configs/$r.yaml ] || { A ALERT "$r: no config, skipped"; continue; }
  echo "[n54 $(date '+%m-%d %H:%M:%S')] 5080 chain: $r" >> $LOG
  for try in 1 2 3; do
    GPU=1 FREE_MIN=1 UNTIL="$U" bash temp/run_5080_sched.sh $r
    [ -f models/checkpoints/$r/DONE ] && break
    [ $(date +%s) -ge $((END - 1500)) ] && { A WARN "$r paused at the end of the 5080 window ($(grep '^step' logs/$r.log | tail -1 | cut -c1-30))"; exit 0; }
    err=$(grep -E 'Error|Traceback|TRIPPED' logs/$r.log | tail -1 | cut -c1-120)
    if [ $try = 3 ]; then A ALERT "$r FAILED 3 times, chain moves on. $err"
    else A WARN "$r attempt $try ended early ($(grep '^step' logs/$r.log | tail -1 | cut -c1-30) $err) -> auto-resume"; sleep 60; fi
  done
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  echo "[n54 $(date '+%m-%d %H:%M:%S')] [$r] evals DONE" >> $LOG
  c1=$(grep "r25-evals $r .*chunk1" $LOG | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
  hay=$(grep "r25-evals $r .*N8:" $LOG | tail -1 | sed -n 's/.*\(N8:.*\)/\1/p' | cut -c1-90)
  A DONE "$r chunk1 ${c1:-?} val $(grep 'eval\] val_ppl' logs/$r.log | tail -1 | awk '{print $3}') | $hay"
done
A DONE "5080 chain finished: the 5080 is idle"
