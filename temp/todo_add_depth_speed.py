p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
old = "<b>For v4:</b> run a 2/3/4L depth ladder signed vs tfmr before trusting the 0.89&times;.</li>"
assert s.count(old) == 1
new = ("<b>For v4:</b> run a 2/3/4L depth ladder signed vs tfmr before trusting the 0.89&times;. "
       "<b>THROUGHPUT (measured 09-30, 5080, post-MADV logs + <code>temp/flops_2L_3L.py</code>):</b> matmul FLOPs/tok (train) "
       "2L signed 274M, 3L 362M (+32%), r13a [spk,attn,attn] 357M, r13d 2L n4096 338M, tfmr3 349M; but wall-clock 2L signed "
       "34-36k tok/s vs 3L 20.3k (<b>-42%, ~1.7&times; slower</b>) &rarr; the spiking block is overhead/memory-bound (LIF FPT K=10, GDN scan), "
       "not FLOP-bound; tfmr3 ~80k tok/s at the same FLOPs. At matched wall-clock the 2L gets ~1.7&times; the tokens &rarr; 3L signed is "
       "NOT worth it at 30M unless it beats 2L-with-1.7&times;-tokens. Adding depth as ATTENTION (r13a) should cost far less wall time; "
       "record r13 tok/s when they run. Idea: judge all depth/width screens on ppl at matched WALL-CLOCK, not matched tokens.</li>")
s = s.replace(old, new)
open(p, "w").write(s)
print("[OK] to-do updated")
