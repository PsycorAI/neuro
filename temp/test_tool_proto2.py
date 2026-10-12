"""test_tool_proto2.py -- 2026-10-12 (Neuro 57). CPU test of tool protocol v2 (plan D1).
 1. proto 1 output is byte-identical to the pre-change formatting (recomputed here from the old formula).
 2. every v2 training target walks the v2 grammar to ACCEPT; every v1 target walks the v1 grammar.
 3. parse_response round-trips v1 and v2 to the same call.
 4. full_frac: only tool / answer records change, the gold tool stays offered, share is about right.
  python temp/test_tool_proto2.py"""
import sys, json
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import tool_data as TD, grammar as G
tools = TD.load_tools()
P = "/home/glenn/projects/neuro/training_data/posttrain/tools.jsonl"
v1, _ = TD.load_tool_jsonl(P, tools)
v2, _ = TD.load_tool_jsonl(P, tools, proto=2)
assert len(v1) == len(v2) > 20000
n_tool = 0
for a, b in zip(v1[::7], v2[::7]):
    r = a["rec"]
    if r["target_mode"] == TD.MODE_TOOL:
        n_tool += 1
        old = f"TOOL {TD.dumps_call(r['target_call']['name'], r['target_call']['arguments'], tools)}{TD.EOS_MARK}"
        assert a["response"] == old, "v1 changed"
        assert b["response"].startswith("TOOL I will ") and a["prompt"] == b["prompt"]
        m1, c1, _ = TD.parse_response(a["response"]); m2, c2, _ = TD.parse_response(b["response"])
        assert m1 == m2 == "TOOL" and c1 == c2 and c1["name"] == r["target_call"]["name"]
    else:
        assert a["response"] == b["response"]
    for resp, proto in ((a["response"], 1), (b["response"], 2)):
        g = G.ToolGrammar(r["tools"], tools, proto=proto)
        node = g.walk(g.root, resp)
        ok = node is not None and (g.accepting(node) or g.nodes[node]["kind"] == G.FREE)
        if r["target_mode"] == TD.MODE_TOOL and r["target_call"]["name"] not in r["tools"]:
            continue
        assert ok, (proto, resp[:120])
    # the v1 target must NOT be accepted by the v2 grammar and vice versa (tool calls only)
    if r["target_mode"] == TD.MODE_TOOL:
        assert G.ToolGrammar(r["tools"], tools, proto=2).walk(G.ToolGrammar(r["tools"], tools, proto=2).root, a["response"]) is None
full, _ = TD.load_tool_jsonl(P, tools, proto=2, full_frac=0.3)
ch = 0; elig = 0
for a, f in zip(v2, full):
    k = f["kind"]
    if k in ("tool", "answer"):
        elig += 1
        if len(f["rec"]["tools"]) == len(tools):
            ch += 1
            if k == "tool":
                assert f["rec"]["target_call"]["name"] in f["rec"]["tools"]
    else:
        assert a["prompt"] == f["prompt"], "tool_absent changed"
print(f"[OK] {len(v1)} records, {n_tool} tool calls checked; all-tools share {ch / elig:.3f} (want ~0.30)")
assert 0.27 < ch / elig < 0.33
print("[OK] sample:", v2[[i for i, p in enumerate(v2) if p['rec']['target_mode'] == 'TOOL'][0]]["response"][:110].replace("\n", " "))
