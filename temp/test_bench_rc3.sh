#!/bin/bash
# test_bench_rc3.sh -- 2026-10-11 (Neuro 55): control-flow test of cloud/bench_rc3.sh with the REAL cloud configs and a
# fake "python" (no GPU). The fake profiler "runs out of memory" at B16 unless mqar_separate_backward is set, and for
# the conversion also at B16 with it and at B8 without it. Expected picks with BG="16:1:s 8:2 4:4":
#   dense 16:1:s (separate backward), conversion 4:4 (plain) -> mk_v4_cloud.py ... 16 1 4 d
# The aux batches must follow B, the stand-in data files must be used (the cloud data paths do not exist at home), and
# the budget block runs under the real python. Then the printed mk_v4_cloud command is run into a scratch copy.
cd /home/glenn/projects/neuro
cat > /tmp/fakepy_bench <<'EOF'
#!/bin/bash
if [ "$1" = "-" ]; then exec /home/glenn/projects/bdh/venv/bin/python "$@"; fi
tag=$(echo "$*" | sed -n 's/.*--tag \([a-z_]*\).*/\1/p'); B=$(echo "$*" | sed -n 's/.*--batch \([0-9]*\).*/\1/p')
sep=0; echo "$*" | grep -q 'mqar_separate_backward=true' && sep=1
echo "ARGS: $*"
oom=0
case $tag in dense_*) [ "$B" = 16 ] && [ $sep = 0 ] && oom=1;;
             conv_*)  [ "$B" = 16 ] && oom=1; [ "$B" = 8 ] && [ $sep = 0 ] && oom=1;; esac
if [ $oom = 1 ]; then echo "torch.OutOfMemoryError: CUDA out of memory (fake)"; exit 1; fi
case $tag in dense_plain) v=34,000;; dense_gap) v=30,000;; dense_xchunk) v=24,000;; conv_gap) v=22,000;; conv_xchunk) v=18,000;; esac
echo "[prof $tag] median step 1000 ms = $v tok/s (B${B}, 12 steps)"
echo "[prof $tag] peak allocated 100.0 GB | peak reserved 101.0 GB"
EOF
chmod +x /tmp/fakepy_bench
mv logs/bench_rc3.log /tmp/bench_rc3.keep 2>/dev/null
ROOT=/home/glenn/projects/neuro PY=/tmp/fakepy_bench BG="16:1:s 8:2 4:4" bash cloud/bench_rc3.sh | grep -v missing | cut -c1-220
echo "--- what the profiler was asked to run (xchunk arms)"
grep -h '^ARGS' logs/bench_dense_xchunk.log logs/bench_conv_xchunk.log | sed 's/.*--batch/--batch/' | cut -c1-330
cmd=$(grep 'margin 7%' logs/bench_rc3.log | sed -n 's/.*->  python temp\/mk_v4_cloud.py //p')
echo "--- generator with the printed arguments: $cmd"
mkdir -p /tmp/cfgscratch && cp configs/v4_cloud.yaml configs/v4_cloud_conv.yaml configs/v4_reh.yaml configs/v4_reh_conv.yaml configs/v4_smoke1b.yaml configs/v4_smoke1b_conv.yaml /tmp/cfgscratch/
/home/glenn/projects/bdh/venv/bin/python temp/mk_v4_cloud.py $cmd | head -n 1
grep -HE '^(batch_size|grad_accum|mqar_batch|mqar_aug_batch|mqar_separate_backward|max_tokens):' configs/v4_cloud.yaml configs/v4_cloud_conv.yaml | sed 's#configs/##' | tr '\n' ' '; echo
cp /tmp/cfgscratch/*.yaml configs/ && echo "(configs restored)"; (cd /tmp/cfgscratch && md5sum *.yaml) > /tmp/cfg_restore.md5; (cd configs && md5sum -c /tmp/cfg_restore.md5 | grep -c OK)
rm -rf /tmp/cfgscratch logs/bench_rc3.log logs/bench_dense_*.log logs/bench_conv_*.log /tmp/fakepy_bench; mv /tmp/bench_rc3.keep logs/bench_rc3.log 2>/dev/null; true
