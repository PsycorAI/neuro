#!/bin/bash
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_p26_configs.py | grep p26v
grep -E "^(mqar_xchunk_scatter|lr_schedule|lr):" configs/screen_p26v_7L_polish_rw_off.yaml
$PY temp/mk_polish_seed.py models/checkpoints/screen_p26e_7L_xscatter/step_0006104.pt models/checkpoints/screen_p26v_7L_polish_rw_off configs/screen_p26v_7L_polish_rw_off.yaml
setsid -f bash temp/queue_5080_p26.sh screen_p26v_7L_polish_rw_off > /dev/null 2>&1
sleep 1; pgrep -af "queue_5080|queue_spk"
echo "[n53 $(date +%m-%d\ %H:%M)] rewarmed polish p26q 76.44 / p26r (+KD) 75.50 = no better than tail polish (75.05/74.72): ppl cost does not come back while scatter stays on. Queued p26v (rewarmed polish, scatter OFF) after p26u." >> logs/gap_queue.log
