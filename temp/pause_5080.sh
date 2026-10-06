#!/bin/bash
# pause_5080.sh -- cleanly free the 5080 (2026-09-25, Glenn needs it ~19:00-00:00).
# 1) stop the queued "resume depth after indepwin" waiter and the screen runners
#    so nothing auto-starts the next arm; 2) SIGINT the 5080 trainer -> train_gpu
#    finishes its step, saves a FINAL ckpt, exits 130 (no .done marker -> resumable);
# 3) wait until the 5080 has no compute process. Resume: temp/resume_5080.sh.
LOG=/home/glenn/projects/neuro/logs/pause_5080.log
say(){ echo "[pause $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "pausing 5080 screens"
pkill -f 'while pgrep -f run_5080_indepwin' && say "stopped queued depth-resume waiter"
pkill -f 'temp/run_5080_(indepwin|depth)\.sh' && say "stopped screen runner(s)"
pids=$(pgrep -f 'train_gpu.py --config configs/screen_(depth|indepwin)_' ; pgrep -f 'recall_eval.py --runs screen_bs512')
if [ -n "$pids" ]; then
  say "SIGINT -> $pids (final ckpt save)"
  kill -INT $pids
  for i in $(seq 1 60); do
    alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break; sleep 10
  done
  [ $alive -eq 1 ] && { say "still alive after 10 min -> SIGTERM"; kill $pids; }
fi
say "done. 5080: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
tail -n 3 /home/glenn/projects/neuro/logs/indepwin_screen.log | tee -a "$LOG"
