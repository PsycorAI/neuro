#!/bin/bash
# r16 build step 1: back up the sources, then save the flags-off parity reference (CPU only).
cd /home/glenn/projects/neuro
for f in model hybrid_attn engram_v2 train_gpu; do cp -n src/$f.py temp/$f.py.bak_r16; done
ls -la temp/*.bak_r16
CUDA_VISIBLE_DEVICES="" /home/glenn/projects/bdh/venv/bin/python temp/parity_r16_flags.py save
