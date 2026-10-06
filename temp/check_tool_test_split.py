# check_tool_test_split.py -- is tool_eval's first-200 test slice all gold-ANSWER? (2026-09-26)
import sys, collections
sys.path.insert(0, "src")
import tool_data
tools = tool_data.load_tools()
pairs, bad = tool_data.load_tool_jsonl("training_data/posttrain/tools.jsonl", tools, split="test")
print("test total", len(pairs), collections.Counter(p["rec"]["target_mode"] for p in pairs))
print("first 200", collections.Counter(p["rec"]["target_mode"] for p in pairs[:200]))
print("first 150 kinds", collections.Counter(p["rec"].get("kind") for p in pairs[:150]))
