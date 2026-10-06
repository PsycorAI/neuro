#!/bin/bash
# Download rate of the in-flight FineWeb-Edu shard (corpus v4 build).
D=/home/glenn/projects/bdh/data/tokenized_v4/dl
f=$(find $D -name '*.incomplete' | head -1)
if [ -z "$f" ]; then echo "no shard downloading"; ls -la $D/sample/100BT 2>/dev/null; tail -n 2 /home/glenn/projects/neuro/logs/build_corpus_v4.log; exit; fi
a=$(stat -c %s "$f"); sleep 30; b=$(stat -c %s "$f")
echo "rate $(( (b - a) / 30 / 1024 )) KB/s; downloaded $(( b / 1024 / 1024 )) MB"
