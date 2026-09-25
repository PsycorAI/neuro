#!/bin/bash
# Interactive REPL with a PsycorNeuro checkpoint.
# Type your prompt, hit Enter to send. Empty line to quit.
#
# Usage:
#   ./chat.sh                                          # sft_500m_full_v3 (best chat template)
#   ./chat.sh dpo_500m_standard_v3_preserved           # DPO'd variant
#   ./chat.sh sparsity_cleanup2_500m_v3                # freshest but degrades chat quality
#   ./chat.sh sft_500m_full_v3 --temperature 1.0       # override sampling params
cd "$(dirname "$0")"
source /home/glenn/projects/bdh/venv/bin/activate
python scripts/chat.py --run "${1:-sft_500m_full_v3}" "${@:2}"
