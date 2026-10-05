#!/bin/bash
# queue_5080_r25.sh -- 2026-10-04 23:10 (Neuro 51). 5080 lent by Glenn until 18:00 10-05 (UNTIL mode: no start or eval
# after 17:35, SIGINT pause at 17:55, then exit). Waits for the r23b queue (GPU-1 lock), then the r25 6L proxies
# (temp/mk_r25_configs.py), each followed by the r25 beyond-window evals. Control = r23b (also gets the r25 evals, chunk 512).
# Launch: setsid -f bash temp/queue_5080_r25.sh > /dev/null 2>&1      WATCH: tail -f logs/gap_queue.log
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
UNTIL_AT="2026-10-05 18:00"
END=$(( $(date -d "$UNTIL_AT" +%s) - 1500 ))      # last start / eval time (17:35)
say(){ echo "[s1 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
r25evals(){   # $1 = run, $2 = forced chunk ("" = model's own), $3 = T
  [ "$(date +%s)" -ge $END ] && { say "[$1] r25 evals SKIPPED (window ends)"; return; }
  local ck; ck=$(ls models/checkpoints/$1/step_*.pt 2>/dev/null | tail -1)
  [ -z "$ck" ] && return
  GPU=1 CHUNK=$2 ITERS=20 bash temp/run_r25_evals.sh "$1=$ck" 2>&1 | sed "s/^/  r25-evals /" >> "$LOG"
}
flock logs/gap_queue_gpu1.lock true            # r23b (queue_5080_r23.sh) finishes first
# 10-05 03:40 relaunch (p25a: memory used but recall never emerged -> fix = plain MQAR batch 32 + xchunk from 30%):
# p25f (running under the previous loop) gets its r25 evals here, then the fixed runs.
r25evals screen_p25f_6L_chunk512_x_nl ""
for r in screen_p25g_6L_chunk512_x_fix screen_p25b_6L_chunk512 screen_p25h_6L_chunk512_x_nl_fix \
         screen_p25e_6L_chunk512_x_H3 screen_p25d_6L_chunk512_T2048_x screen_p25c_6L_chunk256_x; do
  [ "$(date +%s)" -ge $END ] && { say "[$r] not started (window ends $UNTIL_AT)"; break; }
  GPU=1 UNTIL="$UNTIL_AT" bash temp/run_5080_sched.sh $r
  if [ ! -f models/checkpoints/$r/DONE ]; then        # crashed / killed by the watchdog -> next run (window allowing)
    say "[$r] not DONE -> skipping its evals"; continue
  fi
  r25evals $r ""
  case $r in *T2048*) [ "$(date +%s)" -lt $END ] && CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python \
      temp/chunk_eval.py --T 2048 --iters 10 "$r=$(ls models/checkpoints/$r/step_*.pt | tail -1)" 2>&1 \
      | grep 'C=' | sed "s/^/  r25-evals /" >> "$LOG";; esac
done
say "r25 5080 queue DONE"
