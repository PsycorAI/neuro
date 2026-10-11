#!/bin/bash
# push_from_home.sh <ssh-host> <ssh-port> -- run in WSL at home (2026-10-04 draft). Copies code, configs, the vocab map
# and the uint16 v4 corpus (46 GB, LUT already applied -> data_prelut: true) to /workspace on the Vast instance.
# rsync is resumable: re-run it if the upload is interrupted.
set -e
H=${1:?ssh host}; P=${2:?ssh port}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new"
cd /home/glenn/projects/neuro
$SSH root@$H "mkdir -p /workspace/neuro/data /workspace/data/tokenized_v4 /workspace/neuro/logs /workspace/neuro/models/checkpoints"
rsync -az --info=progress2 -e "$SSH" src scripts cloud root@$H:/workspace/neuro/
$SSH root@$H "mkdir -p /workspace/neuro/temp"
# 2026-10-11: + the eval scripts (the memory go/no-go has to be read ON the instance: a 1B checkpoint does not fit a
# home GPU with these evals) and their runner. There: PY=/workspace/venv/bin/python T=2048 bash temp/run_r25_evals.sh name=ckpt
rsync -az -e "$SSH" temp/profile_train.py temp/haystack_recall_eval.py temp/mk_spike_seed.py \
    temp/chunk_eval.py temp/xchunk_haystack_eval.py temp/nl_fact_eval.py temp/xcopy_eval.py temp/multichunk_eval.py \
    temp/xcopy_sweep_eval.py temp/energy_eval_v2.py temp/energy_eval.py temp/run_r25_evals.sh \
    root@$H:/workspace/neuro/temp/
rsync -az -e "$SSH" configs/v4_cloud.yaml configs/v4_cloud_conv.yaml configs/v4_cloud_bench.yaml \
    root@$H:/workspace/neuro/configs/
# 2026-10-11: TEACHER=1 also sends the gated Llama-3.2-1B teacher from the home HF cache (2.4 GB, ~4 h on this line; only
# the conversion phase needs it). Faster: HF_TOKEN=... bash cloud/get_teacher.sh on the instance, which also checks it.
if [ "${TEACHER:-0}" = 1 ]; then
  $SSH root@$H "mkdir -p /workspace/hf/hub"
  rsync -a --partial --info=progress2 -e "$SSH" /home/glenn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B \
      root@$H:/workspace/hf/hub/
fi
rsync -az -e "$SSH" data/vocab_map_32768.npy root@$H:/workspace/neuro/data/
# 2026-10-11: small stand-in data (temp/mk_bench_data.py, ~210 MB = ~20 min) so cloud/bench_rc3.sh can run before the
# Drive download and before the 7.5 GB KD slice is up. BENCH_DATA=0 skips it.
if [ "${BENCH_DATA:-1}" = 1 ] && [ -f data/bench_u16.bin ]; then
  rsync -a --partial --info=progress2 -e "$SSH" data/bench_u16.bin data/bench_u32.bin root@$H:/workspace/neuro/data/
fi
# 2026-10-09: the corpus no longer goes over this DSL line (46 GB = ~80 h). Glenn uploads the pieces to Google Drive;
# on the instance run: bash /workspace/neuro/cloud/pull_corpus.sh
echo "[push] done -- next on the instance: setup_instance.sh, pull_corpus.sh, get_teacher.sh, bench_rc3.sh (cloud/README.md)"
