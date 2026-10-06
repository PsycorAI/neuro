#!/bin/bash
# fix_2055.sh -- 2026-10-05: r25l spilled (33.5 GB, 3k tok/s). Kill, regen config as B8xGA2, resume from its latest ckpt
# (same tokens/step), with evals after; watchdog relaunched (it exits when no queue is alive, so keep this one alive).
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/queue_5090_r25l.sh"); do kill $p; done
for p in $(pgrep -f "configs/r25l_late_insH6_150m.yaml"); do kill -TERM $p; done
sleep 5
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py | grep r25l
grep -E '^(batch_size|grad_accum|mqar_aug_batch|mqar_batch|mqar_xchunk_batch):' configs/r25l_late_insH6_150m.yaml | tr '\n' ' '; echo
cat > temp/queue_5090_r25l2.sh <<'EOS'
#!/bin/bash
# queue_5090_r25l2.sh -- resume r25l (B8xGA2) + evals.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python; r=r25l_late_insH6_150m
echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] RESUME as B8xGA2 (spilled at B16)" >> $LOG
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY src/train_gpu.py --config configs/$r.yaml --device cuda >> logs/$r.log 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] exit=$? :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 | sed "s/^/  recall /" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PY temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25l queue DONE" >> $LOG
EOS
setsid -f bash temp/queue_5090_r25l2.sh > /dev/null 2>&1
sed -i 's/\[r\]un_5080_sched.sh"/[r]un_5080_sched.sh|[q]ueue_5090_r25l2.sh"/' temp/watchdog_tps.sh
setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
sleep 3; ps -eo pid,args | grep -E "bash temp/(queue_|watchdog)|train_gpu" | grep -v grep | cut -c1-100
