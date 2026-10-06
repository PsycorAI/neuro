p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "full 10B run &asymp; 10 days on the 5090."
assert s.count(old) == 1
new = old + (" <b>Glenn 09-28: tfmr baseline = 17L all-attention (param-matched, standard); 1-attn-layer &lsquo;transformer&rsquo; = ABLATION only (TO-DO later: v3 shape with spiking blocks &rarr; plain MLP blocks).</b> "
 "<b>ROUND 5 BUILT (default off, parity-proven on the 5080):</b> lif_alpha (surrogate width), spk_mlp (D3 spiking MLP sub-block, W_down zero-init). "
 "<b>ROUND-5 CONFIGS READY</b> (configs/screen_r5_*, control = full stack 73.44; run order): full_s1, lev8, spkmlp (+4.2M, not param-matched), 3L [spk,attn,spk] (72.3M), 3L_2attn [attn,spk,attn] (74.2M; vs 3L tfmr 59.38), signed, alpha1, alpha4, lev16, engram32k (+12.6M), mqar025 (recipe change &rarr; tfmr would need it too). "
 "~30h of 5080; run: DEADLINE=2400 bash temp/run_5080_gap_list.sh &lt;runs&gt; (5090 busy until ~Thu with v3). "
 "<b>OTHER GAPS (TO-RESEARCH/TO-TEST):</b> (1) TRAINING THROUGHPUT at matched compute: 30M spiking ~11&ndash;13k tok/s vs tfmr ~99k; 1B 4.0k tok/s &rarr; biggest pitch risk (CUDA graphs/whole-model compile with lif_exact; GDN-2 kernel cost of the 1024&times;1024 single-head state); "
 "(2) exact INCREMENTAL DECODE: lif_exact still restarts the membrane at 0 each call &rarr; pass initial membrane state (now easy with the sequential kernel) for honest generation speed/energy claims; "
 "(3) ENERGY accounting with integer spikes (a level-4 spike = up to 4 events; re-derive AC/MAC counts; measure if possible); "
 "(4) LONG-CONTEXT eval beyond 512 (ppl at 2k/4k/8k vs tfmr; constant-memory advantage = the core claim); "
 "(5) DOWNSTREAM tasks at mid scale (v3 leak-trained sat at chance); (6) SFT recall erosion (rehearsal fix, retest on healed base); "
 "(7) honest-comparison hygiene: equal tuning effort per side, 2+ seeds per headline number.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
