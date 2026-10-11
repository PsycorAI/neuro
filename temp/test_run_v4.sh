#!/bin/bash
# test_run_v4.sh -- 2026-10-11 (Neuro 55): logic test of cloud/run_v4.sh with a fake "python" in a fake root under /tmp.
# Fake trainer: dense crashes once (exit 1) then finishes (exit 0); conversion finishes. Fake seed writes a step file.
# A half-written seed (step file without SEEDED) is planted first: run_v4 must redo the seed.
F=/tmp/fake_run_v4; rm -rf $F; mkdir -p $F/configs $F/logs $F/models/checkpoints/d $F/models/checkpoints/c $F/temp $F/src
echo "data_bin: /nonexistent/kd.bin" > $F/configs/c.yaml; : > $F/configs/d.yaml
echo "junk" > $F/models/checkpoints/c/step_0000009.pt            # cut-off seed
cat > $F/fakepy <<'EOF'
#!/bin/bash
case "$1" in
  src/train_gpu.py) run=$(basename $3 .yaml); n=$(cat /tmp/fake_run_v4/n_$run 2>/dev/null || echo 0); echo $((n + 1)) > /tmp/fake_run_v4/n_$run
    echo "fake trainer $run call $((n + 1))"
    touch /tmp/fake_run_v4/models/checkpoints/$run/step_0000010.pt
    [ $run = d ] && [ $n = 0 ] && exit 1
    exit 0 ;;
  temp/mk_spike_seed.py) mkdir -p $3; echo "seeded from $2" > $3/step_0000010.pt; echo "[OK] fake seed"; exit 0 ;;
esac
EOF
chmod +x $F/fakepy
cd /home/glenn/projects/neuro
sed 's/sleep 60/sleep 1/' cloud/run_v4.sh > $F/run_v4_fast.sh
ROOT=$F PY=$F/fakepy HF_HOME=/nonexistent bash $F/run_v4_fast.sh d c; echo "run_v4 exit=$?"
echo "--- logs/run_v4.log"; cat $F/logs/run_v4.log
echo "--- markers"; ls $F/models/checkpoints/d $F/models/checkpoints/c; echo "conv seed content: $(cat $F/models/checkpoints/c/step_0000010.pt)"
echo "--- second start (everything DONE): must do nothing"; ROOT=$F PY=$F/fakepy HF_HOME=/nonexistent bash $F/run_v4_fast.sh d c; tail -n 1 $F/logs/run_v4.log
rm -rf $F
