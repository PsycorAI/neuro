#!/bin/bash
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_p26_configs.py | grep -E "p26[qr]"
SRC=models/checkpoints/screen_p26e_7L_xscatter/step_0006104.pt
for r in screen_p26q_7L_polish_rw screen_p26r_7L_polish_rw_kd; do
  $PY temp/mk_polish_seed.py $SRC models/checkpoints/$r configs/$r.yaml
done
ls models/checkpoints/screen_p26q_7L_polish_rw models/checkpoints/screen_p26r_7L_polish_rw_kd
grep -E "^(lr|muon_lr|lr_schedule|wsd_decay_frac|max_tokens|kd_self_ckpt):" configs/screen_p26r_7L_polish_rw_kd.yaml
setsid -f bash temp/queue_5080_p26.sh screen_p26q_7L_polish_rw screen_p26r_7L_polish_rw_kd > /dev/null 2>&1
sleep 1; pgrep -af "queue_5080|queue_spk"
