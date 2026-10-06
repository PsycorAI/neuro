"""todo_r25b.py -- 2026-10-05 00:00: update the r25 to-do entry (built + queued) and add the follow-up ideas."""
P = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(P).read()
old = "IDEA r25 &mdash; CHUNKED ATTENTION + MEMORY ACROSS CHUNKS (to-build, Neuro 51)</b>: "
new = ("r25 &mdash; CHUNKED ATTENTION + MEMORY ACROSS CHUNKS: <span style=\"color:#1e8449\">BUILT + QUEUED 10-04 23:25</span> "
       "(Neuro 51)</b>: flags <code>attn_chunk</code>, <code>mqar_xchunk_frac</code>/<code>mqar_xchunk_batch</code>, "
       "<code>nlfact_every</code> (+ <code>src/nl_fact_aux.py</code>), all default off (CPU parity + GPU smokes: "
       "<code>temp/smoke_r25_cpu.py</code>, <code>temp/smoke_r25_gpu.py</code>). Evals: <code>temp/run_r25_evals.sh</code> "
       "(chunk_eval, xchunk_haystack_eval, nl_fact_eval). Queue: 5090 r25a (r21b+chunk) / r25b (r21i+chunk) after r21b; "
       "5080 6L proxies p25a/f/b/e/d/c after r23b. Review + ideas: <code>docs/memory_beyond_window_2026_10_05.md</code>. "
       "Baseline r21i with forced chunks: memory &Delta; +0.15 on chunk 2, cross-chunk haystack and NL facts at chance, "
       "in-window NL facts at 200 filler tokens only 28%. ")
assert old in s
s = s.replace(old, new, 1)
anchor = '  <li><b style="color:#2471a3">10-04 22:55'
add = ('  <li><b style="color:#2471a3">10-05 IDEAS from the memory review (docs/memory_beyond_window_2026_10_05.md)</b>: '
       '(E) <b>to-build if p25e helps NL facts</b>: multi-reader H0 (later layers query H0 with contextual queries); '
       '(F) <b>idea</b>: Titans-MAC memory read-out tokens at each chunk start; (G) <b>idea</b>: surprise-gated writes '
       '(&beta; from prediction error; HAM arXiv 2603.22325); (H) <b>to-test in the v4 config</b>: long-chunk final phase '
       '(C 1024&ndash;2048 for the last ~10%); (I) <b>to-build for the demo</b>: chunked streaming decode '
       '(<code>step_kv</code> restarts the cache at chunk boundaries; now guarded with NotImplementedError) + memory '
       'write/read view.</li>\n')
assert anchor in s
s = s.replace(anchor, add + anchor, 1)
open(P, "w").write(s)
print("[OK] to-do updated")
