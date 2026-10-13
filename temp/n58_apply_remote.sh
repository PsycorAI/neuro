#!/bin/bash
# n58_apply_remote.sh -- 2026-10-12 (Neuro 58; Glenn OK 10-12 13:30 "apply the embedding fix after the safety run
# passes"). Runs ON the instance from /workspace/neuro/n58_stage (uploaded by temp/n58_cloud_apply.sh).
# Switches the v4 run to no_decay_names: [embed.weight]:
#   1. checks that the live src files are the launch versions (md5), keeps copies (*.pre_n58)
#   2. installs the staged muon.py / train_gpu.py (compiled first), adds the key to both phase configs
#   3. SIGINT to the trainer: it finishes its step, saves a checkpoint, exits 130; cloud/run_v4.sh restarts it after
#      60 s and it resumes from that checkpoint with the new files (AdamW state remapped).
# Log: logs/n58_apply.log
cd /workspace/neuro || exit 1
S=n58_stage; PY=/workspace/venv/bin/python
{
echo "[apply $(date -u '+%m-%d %H:%M:%S')] start"
[ "$(md5sum < src/muon.py | cut -d' ' -f1)" = b29bc188b1e0e2544bebfd07cd6bd50b ] || { echo "src/muon.py is not the launch version -> abort"; exit 1; }
[ "$(md5sum < src/train_gpu.py | cut -d' ' -f1)" = 329953805b8def9611d5a82f2dfde361 ] || { echo "src/train_gpu.py is not the launch version -> abort"; exit 1; }
$PY -m py_compile $S/muon.py $S/train_gpu.py || { echo "staged files do not compile -> abort"; exit 1; }
pgrep -f "^bash cloud/run_v4.sh" > /dev/null || { echo "cloud/run_v4.sh is not running -> abort (nothing would restart the trainer)"; exit 1; }
pid=$(pgrep -f "python src/train_gpu.py --config configs/v4_cloud.yaml" | head -n 1)
[ -n "$pid" ] || { echo "no dense-phase trainer found -> abort"; exit 1; }
cp -p src/muon.py src/muon.py.pre_n58; cp -p src/train_gpu.py src/train_gpu.py.pre_n58
for f in v4_cloud v4_cloud_conv; do
  cp -p configs/$f.yaml configs/$f.yaml.pre_n58
  grep -q '^no_decay_names:' configs/$f.yaml || printf 'no_decay_names:\n- embed.weight\n' >> configs/$f.yaml
  $PY -c "import yaml,sys; c=yaml.safe_load(open('configs/$f.yaml')); assert c['no_decay_names']==['embed.weight']; print('$f.yaml ok,', len(c), 'keys')" || { echo "config $f broken -> restoring"; cp -p configs/$f.yaml.pre_n58 configs/$f.yaml; exit 1; }
done
cp $S/muon.py src/muon.py; cp $S/train_gpu.py src/train_gpu.py
echo "files installed; last step: $(grep -E '^step' logs/v4_cloud.log | tail -n 1 | cut -c1-80)"
kill -INT $pid; echo "SIGINT -> trainer $pid"
echo "[apply $(date -u '+%m-%d %H:%M:%S')] done; run_v4 restarts the trainer in ~1-3 min"
} >> logs/n58_apply.log 2>&1
tail -n 12 logs/n58_apply.log
