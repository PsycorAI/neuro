#!/bin/bash
# test_bench_rc3.sh -- 2026-10-11 (Neuro 55): control-flow test of cloud/bench_rc3.sh with the REAL cloud configs and a
# fake "python" (no GPU). The fake profiler "runs out of memory" at B16 (both phases) and at B8 for the conversion, so
# the expected picks are dense 8:2 and conversion 4:4, with the aux batches following B and the stand-in data files
# used (the cloud data paths do not exist at home). The budget block runs under the real python.
cd /home/glenn/projects/neuro
cat > /tmp/fakepy_bench <<'EOF'
#!/bin/bash
if [ "$1" = "-" ]; then exec /home/glenn/projects/bdh/venv/bin/python "$@"; fi
tag=$(echo "$*" | sed -n 's/.*--tag \([a-z_]*\).*/\1/p'); B=$(echo "$*" | sed -n 's/.*--batch \([0-9]*\).*/\1/p')
echo "ARGS: $*"
if [ "$B" = 16 ] || { [ "$B" = 8 ] && echo "$tag" | grep -q conv; }; then echo "torch.OutOfMemoryError: CUDA out of memory (fake)"; exit 1; fi
case $tag in dense_plain) v=34,000;; dense_gap) v=30,000;; dense_xchunk) v=24,000;; conv_gap) v=22,000;; conv_xchunk) v=18,000;; esac
echo "[prof $tag] median step 1000 ms = $v tok/s (B${B}, 12 steps)"
echo "[prof $tag] peak allocated 100.0 GB | peak reserved 101.0 GB"
EOF
chmod +x /tmp/fakepy_bench
mv logs/bench_rc3.log /tmp/bench_rc3.keep 2>/dev/null
ROOT=/home/glenn/projects/neuro PY=/tmp/fakepy_bench BG="16:1 8:2 4:4" bash cloud/bench_rc3.sh | cut -c1-220
echo "--- what the profiler was asked to run (xchunk arms)"
grep -h '^ARGS' logs/bench_dense_xchunk.log logs/bench_conv_xchunk.log | sed 's/.*--batch/--batch/' | cut -c1-330
rm -f logs/bench_rc3.log logs/bench_dense_*.log logs/bench_conv_*.log /tmp/fakepy_bench; mv /tmp/bench_rc3.keep logs/bench_rc3.log 2>/dev/null
