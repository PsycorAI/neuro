#!/bin/bash
# fix_1330.sh -- self-KD fixed (Liger CE overwrote the logits): rerun p25u on the 5080 after p25v; r25w on the 5090 after r25v.
cd /home/glenn/projects/neuro
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py | grep r25w
cat > temp/queue_kd_reruns.sh <<'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
( while pgrep -f "bash temp/queue_5080_p25uv.sh" > /dev/null; do sleep 60; done
  [ "$(date +%H%M)" -lt 1630 ] && SCALE=1.5 bash temp/queue_spk_conv.sh 1 screen_p25u_7L_spk_kd \
      models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt ) &
( while pgrep -f "bash temp/queue_5090_r25v.sh" > /dev/null; do sleep 60; done
  SCALE=1.5 bash temp/queue_spk_conv.sh 0 r25w_13L_spk_kd_sc15_150m models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt ) &
wait
EOS
setsid -f bash temp/queue_kd_reruns.sh > /dev/null 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] p25s4 ternary: +89% ppl -> dropped. p25u DIVERGED: bug = self-KD read the logits after Liger fused CE overwrote them in place (fixed: KD before CE). Reruns queued: p25u (5080 after p25v), r25w (12L KD+1.5x thr+L8, 5090 after r25v)" >> logs/gap_queue.log
git add src/train_gpu.py configs/r25w*.yaml logs/gap_queue.log; git add -f temp/mk_r25_configs.py temp/fix_1330.sh temp/queue_kd_reruns.sh
git commit -qm 'fix self-KD (compute before Liger fused CE, which overwrites logits in place); queue p25u/r25w reruns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'; git log --oneline -1
