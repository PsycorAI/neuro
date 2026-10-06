p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "RUNNING 5090: full stack, lev4 s1, hyb+lev4 at HALF LR (muon 0.0025 / adam 3e-4)."
assert s.count(old) == 1
new = old + (" <b>PRIORITIES (Glenn 09-28): (1) v3 heal + post-training (demo for his son) &gt; (2) v4 screens incl. ~100M stacked-recipe check + throughput levers &gt; (3) 426M transformer (after, or in GPU gaps).</b> "
 "QUEUED 5090: temp/run_5090_v3_chain.sh (logs/v3_chain.log) = wait for screens (~15:00) &rarr; heal (+1B tok, ~28h, ckpt 2000) &rarr; token check &rarr; archive leak-base sft/orpo ckpts to models/archive_leakbase_2026_09_28 &rarr; "
 "post-training on the HEALED base (sft_init: heal_v3_363m; pipeline guard/recall base = heal_v3_363m). 426M tfmr waiter CANCELLED (config gained v3's watchdog EMA 0.4/tol 0.07). "
 "3-layer pair moved to the 5080 (DEADLINE 11:45 start, done ~14:10). "
 "TFMR vs v3 AUDIT: matched = data, vocab 32768, block 512, 10B tok, 72 seqs/step, warmup/cosine/lr_min, MTP, MQAR, z-loss, rmsnorm/embed_scale, sampler, eval cadence, seed; "
 "architecture-inherent = all-attn 17L (421.5M vs v3 426M), no GDN/LIF/engram/sparsity, final_norm+depth_scaled_init+single-residual; optimizer = each model&rsquo;s own (AdamW 3e-4 vs NorMuon 0.005/5e-4; NorMuon cost the 30M tfmr 1.28&times;) &mdash; 426M AdamW LR untuned (TO-DECIDE: short LR probe). "
 "UNMATCHABLE: v3 = 10B leak-trained + 1B healed tokens (11B) and its attn layer has the double residual &rarr; v3-vs-tfmr is a caveated comparison; the clean publication comparison is v4 (honest from scratch) vs tfmr.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
