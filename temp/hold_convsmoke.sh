#!/bin/bash
# hold_convsmoke.sh -- 2026-10-10 23:55 (Neuro 55). The 1B smoke RERUN passed its dense phase (resume fix works:
# 13.4 / 29.4 GB) but the CONVERSION phase dies at its first forward on the 5090 (device not ready = WDDM cliff: 1B +
# Llama teacher + spike codes at T2048 does not fit 32 GB). Keep the smoke checkpoints (hard links, no extra space)
# before queue_5090_smoke1b_rerun.sh deletes them, and hold r26v / r26w / r26x (commented in the list) so the
# conversion can be tested at a smaller shape first. temp/queue_5090_convsmoke.sh restores the list and runs the chain.
cd /home/glenn/projects/neuro
for d in v4_smoke1b v4_smoke1b_conv; do
  [ -d models/checkpoints/$d ] && [ ! -d models/checkpoints/${d}_keep ] && cp -al models/checkpoints/$d models/checkpoints/${d}_keep
done
sed -i -E 's/^(r26[vwx]_)/#HOLD \1/' temp/queue_5090_n54.list
ls -la models/checkpoints/v4_smoke1b_keep models/checkpoints/v4_smoke1b_conv_keep; tail -n 4 temp/queue_5090_n54.list
