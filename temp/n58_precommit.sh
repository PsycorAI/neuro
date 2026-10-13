#!/bin/bash
# Neuro 58: list untracked temp files newer than the last commit, and scan the commit candidates for secrets.
cd /home/glenn/projects/neuro || exit 1
last=$(git log -1 --format=%ct)
echo "== untracked temp files newer than HEAD =="
git ls-files --others --ignored --exclude-standard temp | while read -r f; do
  [ -f "$f" ] || continue
  m=$(stat -c %Y "$f")
  [ "$m" -gt "$last" ] && printf '%8s  %s\n' "$(stat -c %s "$f")" "$f"
done
echo "== secret scan =="
grep -rnIEi 'token|secret|password|api[_-]?key|client_id|refresh|BEGIN .*PRIVATE' \
  temp/n57_* docs/HANDOFF_2026_10_12*.md docs/demo_v4_plan_2026_10_12.md configs/brain_mech13.yaml \
  configs/sft_v4_reh13_v2*.yaml scripts/brain_quiz.py scripts/demo/build_search_pack.py scripts/demo/tools_exec.py \
  scripts/gen_searchqa_data.py scripts/gen_toolresult_data.py scripts/selfknow_probe.py scripts/toolresult_eval.py \
  | cut -c1-200 | head -60
