#!/bin/bash
# n58_fetch_kd.sh start|status -- 2026-10-12 (Neuro 58; handoff 10-12b action 3, Glenn 13:20: the upload finishes ~16:20).
# Fetches ONLY train_kd_slice_u32.bin (+ its .sha256) from Glenn's Drive corpus folder to the instance, verifies the
# checksum and puts it where configs/v4_cloud_conv.yaml expects it. It does NOT run cloud/pull_corpus.sh (that would
# also fetch new corpus pieces and rebuild the training file under the live run). Detached on the instance.
# Log there: /workspace/neuro/logs/n58_fetch_kd.log
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
case "$1" in
  start) $SSH root@$H 'cat > /workspace/neuro/n58_stage/fetch_kd.py' <<'PYEOF' 2>/dev/null
import gdown, os, hashlib, shutil, sys
P = "/workspace/data/corpus_pieces"; D = "/workspace/data/tokenized_v4"; N = "train_kd_slice_u32.bin"
fs = gdown.download_folder(url="https://drive.google.com/drive/folders/1ryzuUporjQKZ95v_2sJCI1iEDpozZEbh",
                           skip_download=True, quiet=True)
ids = {os.path.basename(f.path): f.id for f in fs}
assert N in ids and N + ".sha256" in ids, "KD slice is not in the folder yet"
for n in (N + ".sha256", N):
    gdown.download(id=ids[n], output=os.path.join(P, n), quiet=True, resume=True)
want = open(os.path.join(P, N + ".sha256")).read().split()[0]
h = hashlib.sha256()
with open(os.path.join(P, N), "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
size = os.path.getsize(os.path.join(P, N))
if h.hexdigest() != want:
    print(f"CHECKSUM FAILED: size {size}, sha {h.hexdigest()[:16]} != {want[:16]}"); sys.exit(1)
shutil.copyfile(os.path.join(P, N), os.path.join(D, N + ".tmp")); os.replace(os.path.join(D, N + ".tmp"), os.path.join(D, N))
print(f"OK: {os.path.join(D, N)} {size} bytes, sha256 verified")
PYEOF
         $SSH root@$H 'cd /workspace/neuro && setsid -f bash -c "/workspace/venv/bin/python n58_stage/fetch_kd.py > logs/n58_fetch_kd.log 2>&1; echo \"[fetch_kd] exit=\$? \$(date -u +%H:%M)\" >> logs/n58_fetch_kd.log" < /dev/null > /dev/null 2>&1; sleep 2; pgrep -af fetch_kd.py | cut -c1-80' 2>/dev/null ;;
  status) $SSH root@$H 'cd /workspace/neuro; tail -n 4 logs/n58_fetch_kd.log | cut -c1-200; ls -la /workspace/data/corpus_pieces/train_kd_slice* /workspace/data/tokenized_v4/train_kd_slice* 2>/dev/null; df -h /workspace | tail -n 1; grep -E "^step" logs/v4_cloud.log | tail -n 1 | cut -c1-90' 2>/dev/null ;;
esac
