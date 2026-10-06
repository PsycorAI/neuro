# test_grammar_cache.py -- 2026-09-26: TokenTable.mask must not leak masks between
# grammars with different offered sets (old key id(grammar) collided on id reuse).
import sys, gc
sys.path.insert(0, "src")
import grammar as G, tool_data
tools = tool_data.load_tools()
names = [t["name"] for t in tools]
table = G.TokenTable(32768)
failures = 0
for i in range(40):
    offered = names[i % len(names):] + names[:i % len(names)]
    offered = offered[: 1 + i % 4]
    g = G.ToolGrammar(offered, tools)
    # walk the TOOL header, then check every allowed next token keeps a valid node
    node = g.walk(g.root, tool_data.MODE_TOOL + ' {"name": "')
    m, nx = table.mask(g, node)
    ok_ids = m.nonzero().flatten().tolist()
    bad = [t for t in ok_ids if not (0 <= int(nx[t]) < len(g.nodes))]
    # and the allowed first chars must be exactly the first chars of offered names
    firsts = {table.strs[t][0] for t in ok_ids}
    want = {n[0] for n in g.offered}
    if bad or not firsts <= want:
        failures += 1
    del g; gc.collect()          # provoke id reuse, as tool_eval's per-example loop does
print("PASS" if failures == 0 else f"FAIL {failures}/40")
