#!/bin/bash
# Tokenize rate of the corpus v4 build over 60 s, plus CPU use of the builder threads.
f=/home/glenn/projects/bdh/data/tokenized_v4/fwe_topup.bin
a=$(stat -c %s $f); sleep 60; b=$(stat -c %s $f)
echo "tokenize rate $(( (b - a) / 4 / 60 / 1000 ))k tok/s; written $(( b / 4 / 1000000 ))M tok"
top -b -n 1 | head -15 | tail -9
