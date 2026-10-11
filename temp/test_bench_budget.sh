#!/bin/bash
# test_bench_budget.sh -- 2026-10-11 (Neuro 55): the budget block of cloud/bench_rc3.sh, run on the home-test arm logs
# (the 5th arm hit the hook's 900 s limit, so conv_gap is stood in by conv_xchunk). Checks the parsing, the arithmetic
# and the printed mk_v4_cloud.py command, with and without a batch split.
cd /home/glenn/projects/neuro
awk "/<<'EOF'/{f=1; next} /^EOF/{f=0} f" cloud/bench_rc3.sh > /tmp/bench_budget.py
cp logs/bench_conv_xchunk.log logs/bench_conv_gap.log
PY=/home/glenn/projects/bdh/venv/bin/python
echo "--- config batch (home test)"; $PY /tmp/bench_budget.py 4.5 0.25 0.7 - -
echo "--- with splits 8:2 / 4:4";  $PY /tmp/bench_budget.py 4.5 0.25 0.7 8:2 4:4
echo "--- hand check"
$PY - <<'EOF'
vp, vg, vx, cg, cx = 9456, 8283, 8431, 3478, 3478
sd = 0.7 / vp + 0.3 * (0.75 / vg + 0.25 / vx); sc = 0.75 / cg + 0.25 / cx; r = 0.29 / 0.71
D = 4.5 * 86400 / (sd + r * sc); print(f"dense {D / 1e9:.2f}B + conv {r * D / 1e9:.2f}B, mean {1 / sd:,.0f} / {1 / sc:,.0f} tok/s")
EOF
rm -f logs/bench_conv_gap.log /tmp/bench_budget.py
