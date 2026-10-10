"""fix_gen_order.py -- 2026-10-09 (Neuro 54). One-shot: my r26p / r26q blocks landed after the r26g2 emit in
temp/mk_p26_configs.py (before r26n, their source, is emitted). Move them to the end of the file."""
p = "/home/glenn/projects/neuro/temp/mk_p26_configs.py"
s = open(p).read()
a = s.index("# 10-09 18:00 (Neuro 54;")
end_mark = '"after step 10000.")\n'
b = s.index(end_mark, a) + len(end_mark)
blk = s[a:b]
s = s[:a] + s[b:]
s = s.rstrip("\n") + "\n" + blk
open(p, "w").write(s)
print("[OK] moved", blk.count("\n"), "lines to the end")
