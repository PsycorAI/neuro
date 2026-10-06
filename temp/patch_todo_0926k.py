# patch_todo_0926k.py -- round-3 sweep results so far + 426M tfmr B8 probe
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "(<code>temp/sftw_lib.sh</code>, <code>logs/tooleval_n200_&lt;arm&gt;.log</code>).</li>"
b = ("(<code>temp/sftw_lib.sh</code>, <code>logs/tooleval_n200_&lt;arm&gt;.log</code>). "
     "<b>So far</b> (recall N32/N64 | n200 free mode/tool/args/refuse): none 34/3 | .915/.893/.697/.923; w0.1 62/25 | pending; w0.2 53/28 | .860/.844/.631/.846; w0.3 56/22 | .885/.844/.615/.885; w0.5 59/25 | .865/.803/.574/.910. "
     "Recall is FLAT from w0.1 to w0.5 &rarr; added <b>w0.05</b> (5080) to find the knee.</li>")
assert s.count(a) == 1
s = s.replace(a, b)
a2 = "tok/s + VRAM probe on the 5080 queued (<code>temp/probe_tfmr426_b{8,4}.yaml</code>). Est. ~12&ndash;15 days for 10B tok (to be replaced by the measured number)."
b2 = ("<b>5080 probe: B8 = 7.8k tok/s</b> (6.9GB alloc / 16.3 reserved, sys peak 15.9GB &mdash; at the edge; max-autotune) &rarr; <b>10B tok &asymp; 15 days</b> (~18 with Glenn&rsquo;s evening pauses). B4 probe pending.")
assert s.count(a2) == 1
s = s.replace(a2, b2)
open(p, "w", encoding="utf-8").write(s)
print("[OK]")
