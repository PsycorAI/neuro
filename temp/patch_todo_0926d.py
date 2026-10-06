# patch_todo_0926d.py -- 1B probe B=12 result + two v4 levers/risks it surfaced
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "  <li><b>1B A&prime; tok/s+VRAM probe</b> built:"
b = ("  <li><b>1B probe B=12 (2026-09-26)</b>: trained 60 steps but only <b>2,112 tok/s</b> &mdash; 12.7GB <i>allocated</i> vs <b>33.7GB reserved</b> &gt; 32GB &rarr; WDDM spill. B=12 NOT viable as-is. max-autotune compile took ~57 min at this shape.</li>\n"
     "  <li><b>v4 lever TO-TEST: <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code></b> &mdash; reserved (33.7GB) is 2.6&times; allocated (12.7GB) at 1B B=12; expandable segments cut allocator fragmentation and may let B=12 fit under the cliff. Not used anywhere in the repo yet. Probe: rerun probe_1b_b12 with it set.</li>\n"
     "  <li><b>v4 RISK: final/periodic ckpt save OOM'd at 1B</b> &mdash; <code>_save_ckpt</code> &rarr; <code>_to_cpu</code> &rarr; <code>torch.empty_like(..., pin_memory=True)</code> raised CUDA OOM (pinned host alloc; 1B optimizer state is ~12GB, WSL RAM cap 24GB). Must fix before a 1B run (unpinned fallback on failure, or stream per-tensor). TO-FIX.</li>\n"
     + a)
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b, 1))
print("[OK]")
