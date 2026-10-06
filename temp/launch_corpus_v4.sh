#!/bin/bash
# Launch the v4 corpus build detached, at lowest CPU and IO priority (the 5080 screens keep priority).
# WATCH: tail -f /home/glenn/projects/neuro/logs/build_corpus_v4.log
cd /home/glenn/projects/neuro
export RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
setsid -f nice -n 19 ionice -c3 /home/glenn/projects/bdh/venv/bin/python scripts/build_corpus_v4.py --target 8.0e9 \
  >> logs/build_corpus_v4.log 2>&1 < /dev/null
sleep 5
ps -eo pid,ni,args | grep "[b]uild_corpus_v4" | cut -c1-140
tail -n 3 logs/build_corpus_v4.log
