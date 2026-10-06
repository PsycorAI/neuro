"""todo_r25.py -- 2026-10-04 22:55: capture the round-25 idea in docs/to-do.html (idea-capture rule), above the 10-04 entries."""
P = "/home/glenn/projects/neuro/docs/to-do.html"
ANCHOR = '  <li><b style="color:#c0392b">10-04 21:32 r21i'
ENTRY = ('  <li><b style="color:#2471a3">10-04 22:55 IDEA r25 &mdash; CHUNKED ATTENTION + MEMORY ACROSS CHUNKS (to-build, Neuro 51)</b>: '
         'attention only inside fixed, ALIGNED chunks (C=512, every layer the same boundaries; train T=1024-2048); the H0 memory '
         'runs over the whole sequence with FULL gradient (no TBPTT detach). Why it should work where SWA (r21c/g) and r24a failed: '
         'with a sliding window, L&times;W multi-hop relay crosses any distance; with aligned chunks NO attention layer can cross a '
         'boundary, so the memory is the only route to the previous chunk at every depth. Every token keeps exactly the attention '
         'context it has at T=512, so the expected ppl cost is ~0 and chunk-2 tokens get memory context for free. Precedents: '
         'Block-Recurrent Transformer, Block-State Transformer, RMT/ARMT (BPTT across segments), Llama-4 chunked attention. '
         'Also: cross-chunk gap MQAR (kv in chunk A, query in chunk B) so the memory is trained on beyond-window recall '
         '(the demo task). Evals: per-chunk ppl mem on/off, cross-chunk haystack, NL fact recall. Weakness to test next: H0 keys '
         'are token-level (induction store) &rarr; paraphrased NL facts need contextual keys (memory at mid-depth / multi-reader).</li>\n')
s = open(P).read()
assert ANCHOR in s and "IDEA r25" not in s
s = s.replace(ANCHOR, ENTRY + ANCHOR, 1)
open(P, "w").write(s)
print("[OK] to-do r25 idea added")
