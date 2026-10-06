# fix_todo_depth.py -- repair line 104 (sed '&' expanded to the matched text)
p = "docs/to-do.html"
L = open(p, encoding="utf-8").read().split("\n")
i = 103
assert "Depth vs width (30M)" in L[i]
M = "<b>Depth vs width (30M, running):</b>"
L[i] = L[i].replace(M + "mdash;", "&mdash;").replace(M + "rarr;", "&rarr;").replace(M + "amp;", "&amp;").replace(M + "ndash;", "&ndash;").replace(M + "times;", "&times;")
assert M not in L[i].split("Original note:")[0], L[i][:300]
open(p, "w", encoding="utf-8").write("\n".join(L))
print(L[i][:330])
