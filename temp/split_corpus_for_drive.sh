#!/bin/bash
# split_corpus_for_drive.sh -- 2026-10-09 (Neuro 53; Glenn: slow upload -> stage the v4 corpus on Google Drive in pieces).
# 2 GiB pieces of the uint16 LUT corpus (an even byte count -> every piece boundary falls between tokens; the first N
# pieces concatenated = a valid shorter corpus), plus vocab map, metadata, SHA256SUMS and a README.
set -e
SRC=/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin
OUT=/mnt/g/neuro_corpus_v4
mkdir -p $OUT
echo "start $(date)" > $OUT/_progress.txt
split -b 2147483648 -d -a 3 $SRC $OUT/train_lut32768_u16.bin.part_
cp /home/glenn/projects/neuro/data/vocab_map_32768.npy /home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.json \
   /home/glenn/projects/bdh/data/tokenized_v4/meta.json $OUT/
cd $OUT && sha256sum train_lut32768_u16.bin.part_* > SHA256SUMS
cat > $OUT/README.txt << 'EOR'
v4 training corpus (uint16 token ids, LUT to the 32,768 vocab already applied; read with data_prelut: true).
Total 23,151,588,203 tokens = 46.3 GB, split into 2 GiB pieces: train_lut32768_u16.bin.part_000 ... (upload in order).
Pieces split between tokens, so the first N pieces concatenated are a valid shorter corpus (training can start early).
NOTE: the last 14,527,194 tokens of the FULL file are the validation split (data_val_tokens) -> a partial file has no
val tail; set data_val_tokens to a small value or use the full file for evals.
Reassemble on the instance:   cat train_lut32768_u16.bin.part_* > train_lut32768_u16.bin
Verify pieces:                sha256sum -c SHA256SUMS
Also here: vocab_map_32768.npy (copy to neuro/data/), train_lut32768_u16.json + meta.json (corpus metadata).
EOR
echo "done $(date)" >> $OUT/_progress.txt
ls -la $OUT | tail -5 >> $OUT/_progress.txt
