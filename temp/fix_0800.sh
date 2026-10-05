#!/bin/bash
# fix_0800.sh -- 2026-10-05: r25a (12L) memory unused even with chunks -> late read (mem_read_layers). Stop r25b and the
# r25c waiter, run r25d on the 5090 now; 5080 chain: (p25g running) -> p25i first.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
for pat in "bash temp/queue_5090_r25c.sh" "bash temp/queue_5090_r25.sh" "bash temp/run_gap_queue.sh r25b" \
           "bash temp/queue_5080_r25.sh"; do
  for p in $(pgrep -f "$pat"); do kill $p; done
done
for p in $(pgrep -f "configs/r25b_chunk_x1e_150m.yaml"); do kill -TERM $p; done
sleep 5
cat > temp/queue_5090_r25d.sh <<'EOS'
#!/bin/bash
# queue_5090_r25d.sh -- 2026-10-05 08:00: r25d (12L chunk + xchunk from 0 + late read [4,8]) + evals; GPU-0 lock.
cd /home/glenn/projects/neuro
flock logs/gap_queue_gpu0.lock true
for r in r25d_chunk_x1e_read_150m; do
  GPU=0 bash temp/run_gap_queue.sh $r
  ck=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1)
  [ -n "$ck" ] && GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
done
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25d queue DONE" >> logs/gap_queue.log
EOS
setsid -f bash temp/queue_5090_r25d.sh > /dev/null 2>&1
sed -i 's/^for r in screen_p25g_6L_chunk512_x_fix screen_p25b_6L_chunk512 screen_p25h_6L_chunk512_x_nl_fix \\$/for r in screen_p25g_6L_chunk512_x_fix screen_p25i_6L_chunk512_x_read screen_p25e_6L_chunk512_x_H3 \\/; s/^         screen_p25e_6L_chunk512_x_H3 screen_p25d_6L_chunk512_T2048_x screen_p25c_6L_chunk256_x; do$/         screen_p25h_6L_chunk512_x_nl_fix screen_p25d_6L_chunk512_T2048_x screen_p25b_6L_chunk512 screen_p25c_6L_chunk256_x; do/' temp/queue_5080_r25.sh
grep -A1 '^for r in' temp/queue_5080_r25.sh
bash -n temp/queue_5080_r25.sh && setsid -f bash temp/queue_5080_r25.sh > /dev/null 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25a (12L chunk, late xchunk) = memory UNUSED (Δ -6 at eval 1 -> 0; chunk1 43.10 vs r21b 38.84). r25b STOPPED; r25d (late read mem_read_layers [4,8], xchunk from 0) started. 5080 chain: p25g (running) -> p25i (late read) -> p25e -> p25h -> p25d -> p25b -> p25c" >> $LOG
sleep 3; ps -eo pid,args | grep -E "bash temp/(queue_|run_)|train_gpu" | grep -v grep | cut -c1-110
