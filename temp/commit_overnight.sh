#!/bin/bash
# Commit the overnight tools (scorecard, KG recall eval, corpus-v4 builder, val_tokens + audit hooks) and docs.
cd /home/glenn/projects/neuro
git add src/data_stream.py src/model.py src/train_gpu.py scripts/build_corpus_v4.py scripts/interp_scorecard.py \
        scripts/kg_recall_eval.py docs/*.md docs/*.html
git commit -q -F - <<'EOF'
Interpretability scorecard, KG recall eval, v4 corpus builder (+ brain review)

- scripts/interp_scorecard.py: per-checkpoint scorecard (selectivity, dead
  units, attention breadth/sinks, act x grad circuit size, memory trace by
  write decomposition, event-driven op model, optional LLM autointerp);
  appends logs/scorecard.jsonl. model.py: eval-only memory audit capture
  (block.audit, default off).
- scripts/kg_recall_eval.py: forward / reverse / 2-hop in-context recall
  under full / attn_off / mem_off. Finding: reverse recall is below chance
  for both our stack and the 3L transformer (in-context reversal curse).
- scripts/build_corpus_v4.py: old corpus + 8.6B de-duplicated FineWeb-Edu
  (sample/100BT) + the old val tail = 23.15B tokens (~51% FineWeb-Edu).
  data_stream.StreamText val_tokens / config data_val_tokens (default off)
  keeps val windows byte-identical to the old corpus.
- docs: v4 brain-principles + videos review, plan-review evidence update,
  to-do (r15/r16 results, 5090 head_conf issue).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
