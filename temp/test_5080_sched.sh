#!/bin/bash
# Sandbox test of run_5080_sched.sh: Python stub trainer (like train_gpu: SIGINT handler -> save -> exit 130),
# fake clock window (PAUSE = next minute), "midnight" = lift the window.
T=/tmp/schedtest; rm -rf $T; mkdir -p $T/logs $T/models/checkpoints
cat > $T/stub.py <<'EOF'
#!/usr/bin/python3
import sys, os, glob, signal, time
if sys.argv[1] != "src/train_gpu.py": print("eval stub"); sys.exit(0)
run = os.path.basename(sys.argv[3])[:-5]; ck = f"models/checkpoints/{run}"; os.makedirs(ck, exist_ok=True)
if glob.glob(f"{ck}/step_*.pt"):
    print("resumed"); open(f"{ck}/step_0000009.pt", "w").close(); print("[eval] val_ppl 42.0"); sys.exit(0)
open(f"{ck}/step_0000001.pt", "w").close()
hit = []; signal.signal(signal.SIGINT, lambda s, f: hit.append(1))
for _ in range(150):
    if hit: open(f"{ck}/step_0000005.pt", "w").close(); print("SIGINT saved", flush=True); sys.exit(130)
    time.sleep(1)
print("[eval] val_ppl 50.0"); sys.exit(0)
EOF
chmod +x $T/stub.py
P=$(( 10#$(date -d '+1 min' +%H%M) ))
sed -e "s#cd /home/glenn/projects/neuro#cd $T#" -e "s#^PY=.*#PY=$T/stub.py#" \
    -e "s#^to_midnight().*#to_midnight(){ say \"\$1 (TEST: lifting window)\"; PAUSE=9999; NOSTART=9999; sleep 2; }#" \
    /home/glenn/projects/neuro/temp/run_5080_sched.sh > $T/sched.sh
GPU=9 PAUSE=$P NOSTART=$P bash $T/sched.sh runA runB
echo "---- runA.log"; cat $T/logs/runA.log; ls $T/models/checkpoints/*
echo "---- relaunch (both DONE -> skip)"; GPU=9 bash $T/sched.sh runA runB
