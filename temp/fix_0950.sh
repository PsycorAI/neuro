#!/bin/bash
# fix_0950.sh -- p25r was already training on the 5080: make p25s wait for it (and give p25r its r25 evals after).
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/queue_spk_conv.sh 1 "); do kill $p; done
for p in $(pgrep -f "bash temp/queue_5080_p25r2.sh"); do kill $p; done
for p in $(pgrep -f "mk_spike_seed.py.*screen_p25s"); do kill $p; done
for p in $(pgrep -f "configs/screen_p25s_7L_spk_conv.yaml"); do kill -TERM $p; done
sleep 2
cat > temp/queue_5080_p25r_s.sh <<'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
while pgrep -f "run_5080_sched.sh screen_p25r" > /dev/null; do sleep 60; done
r=screen_p25r_7L_H3add_T512c256; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
[ -f models/checkpoints/$r/DONE ] && GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
bash temp/queue_spk_conv.sh 1 screen_p25s_7L_spk_conv models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
EOS
setsid -f bash temp/queue_5080_p25r_s.sh > /dev/null 2>&1
sleep 2; ps -eo pid,args | grep -E "bash temp/(queue_|run_)|train_gpu|mk_spike" | grep -v grep | cut -c1-110
