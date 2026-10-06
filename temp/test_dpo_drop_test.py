# test_dpo_drop_test.py -- drop_test removes exactly the test-split prompts, keeps the rest (2026-09-26)
import sys
sys.path.insert(0, "src")
from train_dpo import load_preferences_jsonl
from tool_data import split_of
P = ["training_data/posttrain/why.jsonl", "training_data/posttrain/abstain.jsonl"]
allp = load_preferences_jsonl(P)
tr = load_preferences_jsonl(P, drop_test=True)
n_test = sum(split_of(r["prompt"]) == "test" for r in allp)
assert len(tr) == len(allp) - n_test, (len(tr), len(allp), n_test)
assert not any(split_of(r["prompt"]) == "test" for r in tr)
print(f"PASS all={len(allp)} test={n_test} train={len(tr)}")
