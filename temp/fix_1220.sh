#!/bin/bash
# fix_1220.sh -- 2026-10-06 review: 5090 -> r25v (12L xcopy, from scratch); 5080 after p25s4 -> p25u (KD + 1.5x thr, L8)
# -> p25v (7L xcopy); p25t (spike from scratch) dropped (conversion already works).
cd /home/glenn/projects/neuro
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py | grep -E 'r25v|p25u|p25v'
for p in $(pgrep -f "bash temp/queue_spk_sweep.sh"); do kill $p; done
cat > temp/queue_5090_r25v.sh <<'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
r=r25v_13L_H6_xcopy_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25v queue DONE" >> logs/gap_queue.log
EOS
setsid -f bash temp/queue_5090_r25v.sh > /dev/null 2>&1
cat > temp/queue_5080_p25uv.sh <<'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
while pgrep -f "bash temp/queue_spk_conv.sh 1 " > /dev/null; do sleep 60; done
SCALE=1.5 bash temp/queue_spk_conv.sh 1 screen_p25u_7L_spk_kd models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
r=screen_p25v_7L_xcopy
GPU=1 UNTIL="2026-10-06 18:00" bash temp/run_5080_sched.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1)
[ -f models/checkpoints/$r/DONE ] && GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
echo "[s1 $(date '+%m-%d %H:%M:%S')] p25u/v queue DONE" >> logs/gap_queue.log
EOS
setsid -f bash temp/queue_5080_p25uv.sh > /dev/null 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] REVIEW: r25t (L8, 80M conv) = no ppl cost (chunk1 41.67 vs 41.74), 5.0x energy, memory intact. 6L: L8 > L4; 2x thresholds 9.7x but +35% (20M conv). Real-text recall beyond window FAILS (xcopy 30 vs 28) -> r25v (12L + xcopy aug) on 5090; 5080: p25u (KD + 1.5x thr) -> p25v (7L xcopy); p25t dropped" >> logs/gap_queue.log
sleep 2; ps -eo pid,args | grep -E "bash temp/(queue_|run_|watchdog)|train_gpu" | grep -v grep | cut -c1-100
