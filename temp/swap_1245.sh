#!/bin/bash
# swap_1245.sh -- 2026-10-08 12:45 (Neuro 53). r26i crashed twice on the 5080 ('device not ready'); move it to the 5090.
# Pause r26e (SIGINT -> ckpt), then on the 5090: resume r26i (from step 8000) -> evals -> resume r26e -> evals.
cd /home/glenn/projects/neuro
k(){ for p in $(pgrep -f "$1"); do echo "$2 $p: $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-80)"; kill -$2 $p; done; }
k "^bash temp/queue_5090_chain3.sh$" TERM
k "^bash temp/run_gap_queue.sh r26e_13L_H6_chunk1024_xlate_150m$" TERM
sleep 1
k "train_gpu.py --config configs/r26e_13L_H6_chunk1024_xlate_150m.yaml" INT
for i in $(seq 1 90); do pgrep -f "configs/r26e_13L" > /dev/null || break; sleep 2; done
ls models/checkpoints/r26e_13L_H6_chunk1024_xlate_150m/
cat > temp/queue_5090_r26i_e.sh << 'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
for r in r26i_13L_RC2_hola_rand128_150m r26e_13L_H6_chunk1024_xlate_150m; do
  GPU=0 FREE_MIN=1 UNTIL="2026-10-10 06:00" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval mem_write_probe multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  CUDA_VISIBLE_DEVICES=0 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
done
EOS
setsid -f bash temp/queue_5090_r26i_e.sh > /dev/null 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26i crashed twice on the 5080 (device not ready, 08:12) -> resumed on the 5090 from step 8000; r26e paused (SIGINT) and resumes after. r26g (Llama KD) spilled on the 5090 (watchdog kill at 1.6k tok/s; evals on a partial ckpt = invalid)." >> logs/gap_queue.log
sleep 3; ps -eo pid,args | grep -E "queue_5090|sched|train_gpu" | grep -v grep | grep -v swap_ | cut -c1-120
