#!/bin/bash
# queue_5090_r25.sh -- 2026-10-04 23:10 (Neuro 51; Glenn: queue promising ideas without asking).
# 1. As each run of the live r24 queue finishes (r24a -> r24b -> r21b), run the r25 beyond-window evals on it with forced
#    512-token attention chunks (temp/run_r25_evals.sh; r24b's stateful memory is the interesting one).
# 2. When that queue releases the GPU-0 lock: r25a (r21b + chunk512/T1024 + xchunk MQAR), r25b (r21i + same), each followed
#    by the standard recall/haystack evals (run_gap_queue.sh) and the r25 evals.
# Launch: setsid -f bash temp/queue_5090_r25.sh > /dev/null 2>&1      WATCH: tail -f logs/gap_queue.log
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
say(){ echo "[q0 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
r25evals(){   # $1 = run, $2 = forced chunk ("" = the model's own)
  local ck; ck=$(ls models/checkpoints/$1/step_*.pt 2>/dev/null | tail -1)
  [ -z "$ck" ] && { say "[$1] r25 evals SKIPPED (no ckpt)"; return; }
  GPU=0 CHUNK=$2 bash temp/run_r25_evals.sh "$1=$ck" 2>&1 | sed "s/^/  r25-evals /" >> "$LOG"
}
for r in r24a_swax32_x1e_150m r24b_stateful_x1e_150m r21b_x1_150m; do
  # the queue logs "haystack (gap ...) <run>" after the run's own evals
  until grep -q "haystack (gap 0/64/192/440) $r" "$LOG"; do
    pgrep -f "[r]un_gap_queue.sh" > /dev/null || break
    sleep 120
  done
  grep -q "haystack (gap 0/64/192/440) $r" "$LOG" && r25evals $r 512
done
flock logs/gap_queue_gpu0.lock true        # wait for the r24 queue to release GPU 0
for r in r25a_chunk_x1_150m r25b_chunk_x1e_150m; do
  GPU=0 bash temp/run_gap_queue.sh $r
  r25evals $r ""
done
say "r25 5090 queue DONE"
