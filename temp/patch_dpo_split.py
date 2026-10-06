# patch_dpo_split.py -- 2026-09-26: ORPO trained on the abstain.jsonl test split that
# abstain_eval/head_eval score on. Add opt-in drop_test to load_preferences_jsonl;
# the training call sets it, head_eval (which wants the test pairs) is unchanged.
p = "src/train_dpo.py"
s = open(p, encoding="utf-8").read()
reps = [
    ("def load_preferences_jsonl(paths, pair_types=None):",
     "def load_preferences_jsonl(paths, pair_types=None, drop_test=False):"),
    ("    the abstention set) without regenerating anything.\n    \"\"\"",
     "    the abstention set) without regenerating anything.\n\n"
     "    drop_test: skip the hash-held-out test split (tool_data.split_of on the\n"
     "    prompt) -- training must set it, since abstain_eval/head_eval score there.\n    \"\"\""),
    ("                if pair_types and pt not in pair_types:\n                    continue\n",
     "                if pair_types and pt not in pair_types:\n                    continue\n"
     "                if drop_test and r.get(\"prompt\") and split_of(r[\"prompt\"]) == \"test\":\n"
     "                    continue\n"),
    ("        raw += load_preferences_jsonl(c[\"dpo_jsonl\"], c.get(\"dpo_pair_types\"))",
     "        raw += load_preferences_jsonl(c[\"dpo_jsonl\"], c.get(\"dpo_pair_types\"),\n"
     "                                      drop_test=True)"),
    ("                        make_encoder_pair)\n",
     "                        make_encoder_pair)\nfrom tool_data import split_of\n"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("[OK] patched", p)
