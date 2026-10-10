#!/bin/bash
cd /home/glenn/projects/neuro
date
grep -aE "r26o" logs/gap_queue.log | tail -8 | cut -c1-230
grep -aE "^r26o.*E3 haystack" logs/e123_diag.log | cut -c1-140
ps -eo args | grep "src/train_gpu.py" | grep -v grep | sed "s/.*configs.//; s/.yaml.*//"
