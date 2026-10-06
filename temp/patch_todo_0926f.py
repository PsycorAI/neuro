# patch_todo_0926f.py -- 30M fixed-transformer preview result + open question
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "  <li><b>30M FIXED-transformer preview</b>"
b = ("  <li><b style=\"color:#c0392b\">30M FIXED-transformer preview RESULT (2026-09-26)</b>: <code>screen_tfmrfix_s0_30m</code> (2L d1024 all-attn, attn_single_residual, 58.75M) vs spiking control (59.58M), same 100M tok/data/MQAR aux/seed: "
     "<b>val_ppl 87.8 vs 55.8</b> (spiking &minus;36%); single-pass recall tfmr 38/12/3/3/0 (= chance) vs spiking 100/100/100/59/38. Transformer ran ~99k tok/s vs ~16k (&asymp;6&times; cheaper per token &rarr; at MATCHED COMPUTE it could see ~6&times; the tokens: the fair headline needs both views). "
     "<b>OPEN / BLOCKS Phase 0 #1:</b> the transformer NEVER learned the MQAR aux (logged total loss ~9.0 &rArr; MQAR CE &asymp; ln V) although attention should solve MQAR easily; the block code (RoPE, QK-norm, causal SDPA, pre-norm) reads correct. MQAR-only toy from scratch (<code>temp/diag_tfmr_mqar*.py</code>) is NOT discriminative (spiking doesn&rsquo;t learn it alone either). Suspects: spiking-tuned LR/Muon on attention, untied head, too-short training for induction-head formation at 2 layers. Resolve (e.g. transformer-standard AdamW LR sweep, 4L) BEFORE the 363M fixed baseline, or its recall comparison is meaningless.</li>\n"
     + a)
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b, 1))
print("[OK]")
