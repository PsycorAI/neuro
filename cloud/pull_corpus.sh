#!/bin/bash
# pull_corpus.sh -- run ON the Vast instance (2026-10-09, NOT RUN YET). Downloads whatever v4 corpus pieces are in
# Glenn's public Google Drive folder ("anyone with the link", no login on the instance), verifies them, and builds
#   /workspace/data/tokenized_v4/train_lut32768_u16.bin   (read with data_prelut: true)
#   /workspace/data/tokenized_v4/train_kd_slice_u32.bin   (raw ids for the conversion phase; only needed at the end)
# PARTIAL corpus: pieces split between tokens, so ANY set of whole pieces + val_tail_u16.bin is a valid corpus with
# the same val windows. All 22 pieces -> the exact full file (the val tail is the end of piece 021).
# Mix: pieces 000-013 = the old 7-source mix, 014-021 = the FineWeb-Edu top-up -> a partial set should hold both
# kinds at about 5 : 3, or the early training mix differs from the 150M proofs.
# Re-run it whenever more pieces are up (finished downloads are kept), then restart the trainer (kill -TERM; run_v4.sh
# resumes it): the trainer sees the corpus size only at start. Needs ~100 GB free. NOT while mk_spike_seed runs.
# If Google answers "quota exceeded" for a piece: wait and re-run, or switch to a service account + rclone.
set -u
URL="https://drive.google.com/drive/folders/1ryzuUporjQKZ95v_2sJCI1iEDpozZEbh"
D=/workspace/data/tokenized_v4; P=/workspace/data/corpus_pieces
mkdir -p $D $P /workspace/neuro/data
/workspace/venv/bin/pip install -q gdown
for try in 1 2 3 4 5; do
  /workspace/venv/bin/gdown --folder "$URL" -O $P --continue && break
  echo "[pull_corpus] gdown attempt $try failed, retrying in 5 min"; sleep 300
done
cd $P || exit 1
[ -f SHA256SUMS ] || { echo "[pull_corpus] SHA256SUMS missing"; exit 1; }
n=$(ls train_lut32768_u16.bin.part_* 2>/dev/null | wc -l); want=$(grep -c 'part_' SHA256SUMS)
[ "$n" -ge 1 ] || { echo "[pull_corpus] no pieces yet"; exit 1; }
sha256sum -c --ignore-missing SHA256SUMS || { echo "[pull_corpus] checksum FAILED: delete the bad piece(s) and re-run"; exit 1; }
if [ "$n" = "$want" ]; then
  cat train_lut32768_u16.bin.part_* > $D/train.tmp
  [ "$(stat -c %s $D/train.tmp)" = 46303176406 ] || { echo "[pull_corpus] joined size is wrong"; exit 1; }
else
  sha256sum -c val_tail_u16.bin.sha256 || { echo "[pull_corpus] partial corpus needs val_tail_u16.bin"; exit 1; }
  cat train_lut32768_u16.bin.part_* val_tail_u16.bin > $D/train.tmp
fi
mv $D/train.tmp $D/train_lut32768_u16.bin
old=$(ls train_lut32768_u16.bin.part_* | grep -cE 'part_0(0[0-9]|1[0-3])$')
echo "[pull_corpus $(date '+%m-%d %H:%M')] joined $n of $want pieces ($old old-mix, $((n - old)) FineWeb-Edu) = $(( $(stat -c %s $D/train_lut32768_u16.bin) / 2000000000 )).x B tokens" | tee -a /workspace/neuro/logs/pull_corpus.log
cp -f train_lut32768_u16.json meta.json $D/ 2>/dev/null
cp -f vocab_map_32768.npy /workspace/neuro/data/ 2>/dev/null
if [ -f train_kd_slice_u32.bin ]; then
  sha256sum -c train_kd_slice_u32.bin.sha256 && cp -f train_kd_slice_u32.bin $D/ || echo "[pull_corpus] KD slice checksum FAILED"
else echo "[pull_corpus] KD slice not uploaded yet (only needed for the conversion phase): re-run later"; fi
echo "[pull_corpus] done"; ls -la $D
