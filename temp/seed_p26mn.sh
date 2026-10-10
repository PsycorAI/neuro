#!/bin/bash
cd /home/glenn/projects/neuro
SRC=models/checkpoints/screen_p26e_7L_xscatter/step_0006104.pt
for r in screen_p26m_7L_polish screen_p26n_7L_polish_kd; do
  /home/glenn/projects/bdh/venv/bin/python temp/mk_polish_seed.py $SRC models/checkpoints/$r configs/$r.yaml
done
ls -la models/checkpoints/screen_p26m_7L_polish/ models/checkpoints/screen_p26n_7L_polish_kd/
