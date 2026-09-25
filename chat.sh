#!/bin/bash
# Interactive REPL with a PsycorNeuro checkpoint.
# Type your prompt, hit Enter to send. Empty line to quit.
# Hebbian brain state carries across turns (/reset-brain to clear).
#
# Usage:
#   ./chat.sh                                   # sft_v3_363m (latest ckpt = v3 instruct model)
#   ./chat.sh orpo_v3_363m                      # after the ORPO stage
#   ./chat.sh opd_v3_363m                       # after the OPD stage
#   ./chat.sh sft_v3_363m --temperature 0.7     # override sampling params
#   GPU=1 ./chat.sh                             # pick the GPU (default 0 = 5090)
# NB: the bare base (flagship_v3_363m) is not instruction-tuned; the chat
# template only makes sense on the post-trained runs above.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
source /home/glenn/projects/bdh/venv/bin/activate
python scripts/chat.py --run "${1:-sft_v3_363m}" "${@:2}"
