#!/bin/bash
cd /home/glenn/projects/neuro/models/checkpoints
for d in */; do d=${d%/}; s=$(du -sm "$d" | cut -f1); n=$(ls "$d" | grep -c "\.pt$"); m=$(stat -c %y "$d" | cut -c1-10); echo "$s $m $n $d"; done | sort -k2 > /home/glenn/projects/neuro/temp/ckpt_inventory.txt
awk "{s+=\$1} END {print s/1024 \" GB in \" NR \" dirs\"}" /home/glenn/projects/neuro/temp/ckpt_inventory.txt
