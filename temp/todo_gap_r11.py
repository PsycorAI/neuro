p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "(7) honest-comparison hygiene: equal tuning effort per side, 2+ seeds per headline number."
assert s.count(old) == 1
new = old + (" <b style=\"color:#c0392b\">!!! RECALL UNDER INTERFERENCE (09-28, temp/haystack_recall_eval.py, logs/haystack_recall_r4.log):</b> "
 "[k1 v1..kN vN] + g tokens of REAL TEXT + [kq]; among-cands acc N8 / N32 at gap 0/64/192/440: "
 "lif_exact 100/96/73/40 | 100/87/62/28; lev4 100/66/55/23 | 100/34/18/8; hybrid 100/14/14/13 | 100/3/2/4; <b>FULL STACK 100/8/10/15 | 100/2/2/3</b>; "
 "hyb3_lev4 100/57/35/19 | 100/34/14/8; tfmr2 chance; tfmr3 100/75/70/44 | 100/84/66/25. "
 "&rarr; our &lsquo;100% recall&rsquo; holds ONLY for immediate queries; the ppl levers (hybrid, lev4) DESTROYED recall robustness; plain lif_exact &asymp; 3L tfmr. "
 "Cause (arXiv 2609.16183 &lsquo;interference wall&rsquo;): MQAR aux has NO gap/distractors. "
 "<b>BUILT (default off, parity-proven): mqar_gap_max / mqar_gap_ramp_frac / mqar_batch (haystack MQAR distance curriculum, private RNG), canon_a (Canon-A, Allen-Zhu Physics of LM 4.1).</b> "
 "Haystack eval now standard in temp/run_gap_queue.sh. ROUND-6 CONFIGS: screen_r6_gap_s0_30m (full stack + haystack MQAR), screen_r6_gap_lifexact_s0_30m, screen_r6_canon_s0_30m. "
 "QUEUE PRIORITY (next free GPU): r6_gap, r6_gap_lifexact, r5_full_s1, r6_canon, r5_3L, r5_spkmlp, r5_lev16/signed/alpha, r5_3L_2attn, r5_engram32k, r5_mqar025. "
 "Headline recall numbers MUST include the haystack eval from now on.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
