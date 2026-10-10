#!/bin/bash
for i in $(seq 1 70); do grep -q "^done" /mnt/g/neuro_corpus_v4/_progress.txt 2>/dev/null && break; sleep 30; done
cat /mnt/g/neuro_corpus_v4/_progress.txt
ls /mnt/g/neuro_corpus_v4 | grep -c part_
wc -l < /mnt/g/neuro_corpus_v4/SHA256SUMS 2>/dev/null
cat /home/glenn/projects/neuro/logs/split_corpus.log
exit 0
