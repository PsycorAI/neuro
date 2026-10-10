#!/bin/bash
# queue_5080_reh.sh -- 2026-10-09 (Neuro 54; Glenn: 5080 free until 17:30 10-10). DRESS REHEARSAL of the cloud pipeline
# at 7L on the 5080: the real cloud/run_v4.sh (dense on the uint16 prelut corpus -> mk_spike_seed -> conversion with
# Llama-3.2-1B KD on the raw uint32 slice, val_ppl tripwire armed) on configs/v4_reh*.yaml, then the cloud evals.
# Goal = find pipeline bugs before the H200, not a quality number. Reports through temp/alert.sh.
# Launch: setsid -f bash temp/queue_5080_reh.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
PYB=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log; A(){ bash temp/alert.sh "$@"; }
S=/home/glenn/projects/bdh/data/tokenized_v4/train_kd_slice_u32.bin
# Glenn stages the Drive upload in batches: files not yet uploading sit in G:\neuro_corpus_v4_backup.
G=$(ls /mnt/g/neuro_corpus_v4/train_kd_slice_u32.bin /mnt/g/neuro_corpus_v4_backup/train_kd_slice_u32.bin 2>/dev/null | head -1)
[ -f $S ] || { [ -n "$G" ] && cp "$G" $S.tmp && mv $S.tmp $S; } || { A ALERT "rehearsal: KD slice copy failed"; exit 1; }
echo "[n54 $(date '+%m-%d %H:%M:%S')] 5080: cloud dress rehearsal v4_reh -> v4_reh_conv (cloud/run_v4.sh)" >> $LOG
A INFO "cloud dress rehearsal started on the 5080"
CUDA_VISIBLE_DEVICES=1 ROOT=/home/glenn/projects/neuro PY=$PYB HF_HOME=/home/glenn/.cache/huggingface \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 bash cloud/run_v4.sh v4_reh v4_reh_conv
vd=$(grep 'eval\] val_ppl' logs/v4_reh.log | tail -1 | awk '{print $3}')
vc=$(grep 'eval\] val_ppl' logs/v4_reh_conv.log 2>/dev/null | tail -1 | awk '{print $3}')
if [ ! -f models/checkpoints/v4_reh_conv/DONE ]; then
  A ALERT "rehearsal FAILED (dense val ${vd:-none}, conv val ${vc:-none}): $(tail -n 1 logs/run_v4.log | cut -c1-120)"; exit 1
fi
ck=$(ls models/checkpoints/v4_reh_conv/step_*.pt | tail -1)
GPU=1 T=2048 bash temp/run_r25_evals.sh "v4_reh_conv=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
CUDA_VISIBLE_DEVICES=1 $PYB temp/energy_eval_v2.py "v4_reh_conv=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
c1=$(grep "r25-evals v4_reh_conv .*chunk1" $LOG | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
en=$(grep "v4_reh_conv energy-v2" $LOG | tail -1 | sed -n 's/.*compute only \([0-9.x]*\).*SRAM fetch \([0-9.x]*\).*/\1 \/ \2 SRAM/p')
A DONE "rehearsal PASSED end to end: dense val $vd -> conv val $vc, chunk1 ${c1:-?}, energy ${en:-?} (7L, 56M tok; pipeline check, not a quality number)"
