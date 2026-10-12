#!/bin/bash
# Neuro 57 (plan D2): the offline assets of the tool executors. sentencepiece (opus-mt tokenizer), WordNet data,
# 5 opus-mt translation models (~300 MB each, into the HF cache). Log: logs/n57_tool_assets.log. Safe to re-run.
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python
{
echo "== $(date) start"
$PY -m pip install -q sentencepiece sacremoses 2>&1 | tail -n 2
$PY -c "import nltk; print('wordnet', nltk.download('wordnet', quiet=True))"
for m in opus-mt-en-es opus-mt-en-fr opus-mt-en-de opus-mt-en-it opus-mt-tc-big-en-pt; do
  timeout 3600 $PY -c "from huggingface_hub import snapshot_download as d; print(d('Helsinki-NLP/$m', allow_patterns=['*.json','*.spm','pytorch_model.bin','*.txt']))" 2>&1 | tail -n 1
done
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= $PY temp/test_tools_exec.py 2>&1 | grep -v -i warning | cut -c1-300 | tail -n 8
echo "== $(date) end"
} > logs/n57_tool_assets.log 2>&1
bash temp/alert.sh DONE "tool assets: $(grep -c snapshots logs/n57_tool_assets.log)/5 translation models | $(grep -E '^\[(OK|FAIL)\]' logs/n57_tool_assets.log | tail -1) | logs/n57_tool_assets.log"
