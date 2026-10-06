"""mk_r25_configs.py -- 2026-10-04 (Neuro 51; Glenn: queue promising long-attention+memory ideas without asking).
r25 = CHUNKED ATTENTION + MEMORY ACROSS CHUNKS. Attention only inside aligned C-token chunks (attn_chunk; every layer has
the same boundaries, so no multi-hop relay can cross one -- unlike the sliding windows of r21c/g and r24a); the H0 memory
runs over the whole block_size sequence with full gradient. Each token keeps the attention context it has at T=512 and
chunks 2.. gain memory context. mqar_xchunk_frac: part of the gap-MQAR aux puts the kv block in chunk A and the query in
chunk B (only the memory can answer). Same tokens/step as the controls (batch halves when T doubles).
150M (5090), controls in the same queue:  r25a = r21b (late x1) + chunk512/T1024 + xchunk 0.5
                                          r25b = r21i (x1 from step 0) + chunk512/T1024 + xchunk 0.5
6L proxy (5080, 100M tok), control r23b:  p25a chunk512/T1024 + xchunk | p25b chunk512/T1024 (natural text only)
                                          p25c chunk256/T1024 + xchunk | p25d chunk512/T2048 + xchunk (3 boundaries)
                                          p25e = p25a + a 2nd memory layer at depth 3 (contextual keys, for NL facts)"""
import yaml
R = "/home/glenn/projects/neuro/configs/"


def emit(name, src, note, **upd):
    c = yaml.safe_load(open(R + src + ".yaml"))
    T = upd.get("block_size", 1024)
    upd.setdefault("batch_size", c["batch_size"] * c["block_size"] // T)      # same tokens per step
    upd.setdefault("eval_batch", max(1, c["eval_batch"] * c["block_size"] // T))
    # batch <= 16 now -> Inductor SplitScan cumsum crash risk (GPU smoke crashed at B16, d128); same exact math, eager.
    upd.setdefault("gdn2_cumsum_eager_max_b", 16)
    # 10-05 03:30 fix after p25a (recall never emerged): keep the control's plain-MQAR batch; xchunk from 30%
    upd.setdefault("mqar_aug_batch", c["batch_size"])
    if upd.get("mqar_xchunk_frac"):
        upd.setdefault("mqar_xchunk_start_frac", 0.3)
    c.update(upd, run_name=name)
    # 10-05 06:30: the 16 GB 5080 spills with the plain-MQAR batch at 32 (p25g 15.3 GB, 20.5k -> 8k tok/s). 6L proxies
    # that have not run yet use 2 micro-batches with every per-micro aux halved (same samples per optimizer step).
    if name in ("screen_p25g_6L_chunk512_x_fix", "screen_p25i_6L_chunk512_x_read", "screen_p25l_6L_chunk512_x_H3_nl",
                "screen_p25m_7L_chunk512_x_H0H3", "screen_p25n_6L_chunk512_x_H3only", "screen_p25o_7L_chunk512_x_H3add",
                "screen_p25p_7L_H3add_nl", "screen_p25q_7L_H3add_T2048", "screen_p25r_7L_H3add_T512c256", "screen_p25b_6L_chunk512", "screen_p25e_6L_chunk512_x_H3",
                "screen_p25d_6L_chunk512_T2048_x", "screen_p25c_6L_chunk256_x"):
        c.update(batch_size=c["batch_size"] // 2, grad_accum=2, mqar_aug_batch=c["mqar_aug_batch"] // 2,
                 mqar_batch=c["mqar_batch"] // 2, mqar_xchunk_batch=max(1, c.get("mqar_xchunk_batch", 4) // 2))
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r25_configs.py (2026-10-04) from {src}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name, {k: c[k] for k in ("block_size", "batch_size", "eval_batch", "attn_chunk")})


X = dict(mqar_xchunk_frac=0.5)
emit("r25a_chunk_x1_150m", "r21b_x1_150m", "+ attn_chunk 512, T 1024, xchunk MQAR 0.5 (late dose).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, **X)
emit("r25b_chunk_x1e_150m", "r21i_x1early_150m", "+ attn_chunk 512, T 1024, xchunk MQAR 0.5 (dose from step 0).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, **X)
P = "screen_r23b_g3w_6L_x1e"
emit("screen_p25a_6L_chunk512_x", P, "+ attn_chunk 512, T 1024, xchunk MQAR 0.5 (ran 10-05: plain MQAR batch 16, "
     "xchunk from 0).", block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, mqar_aug_batch=16,
     mqar_xchunk_start_frac=0.0, **X)
emit("screen_p25g_6L_chunk512_x_fix", P, "p25a + plain MQAR batch 32 + xchunk from 30% (fix for p25a recall collapse).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, **X)
emit("screen_p25b_6L_chunk512", P, "+ attn_chunk 512, T 1024 (natural text only, no xchunk MQAR).",
     block_size=1024, attn_chunk=512)
emit("screen_p25c_6L_chunk256_x", P, "+ attn_chunk 256, T 1024, xchunk MQAR 0.5.",
     block_size=1024, attn_chunk=256, mqar_xchunk_batch=4, **X)
emit("screen_p25d_6L_chunk512_T2048_x", P, "+ attn_chunk 512, T 2048 (3 boundaries), xchunk MQAR 0.5.",
     block_size=2048, attn_chunk=512, mqar_xchunk_batch=4, **X)
emit("screen_p25e_6L_chunk512_x_H3", P, "+ attn_chunk 512, T 1024, xchunk 0.5, 2nd memory layer at depth 3.",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, attention_layer_positions=[1, 2, 4, 5], **X)
# 10-05 00:00: natural-language fact episodes (src/nl_fact_aux.py), facts in chunk A, question in chunk B (70%)
NL = dict(nlfact_every=2, nlfact_batch=4, nlfact_weight=0.15, nlfact_in_frac=0.3, nlfact_k_max=4)
emit("screen_p25f_6L_chunk512_x_nl", P, "+ attn_chunk 512, T 1024, xchunk 0.5, NL-fact aux (every 2nd step, w 0.15).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, mqar_aug_batch=16, mqar_xchunk_start_frac=0.0,
     **X, **NL)                                   # ran 10-05 03:30 with the pre-fix MQAR settings
# p25f spilled on the 16 GB 5080 (17.1 GB): p25h = 2 micro-batches of 8, every per-micro aux halved (same samples/step)
emit("screen_p25h_6L_chunk512_x_nl_fix", P, "p25f + plain MQAR batch 32 + xchunk from 30%; B8 x GA2 to fit 16 GB.",
     block_size=1024, attn_chunk=512, batch_size=8, grad_accum=2, mqar_aug_batch=16, mqar_batch=4,
     mqar_xchunk_batch=2, **X, **NL)
# 10-05 07:45: r25a (12L) memory Δ -6 at eval 1, then 0 -> the READ path fails at depth. Late read = H0's memory
# read-out re-injected after blocks 4 and 8 (6L: 3) through zero-init projections; xchunk from step 0 (p25a 6L used it).
emit("r25d_chunk_x1e_read_150m", "r21i_x1early_150m", "+ chunk512/T1024 + xchunk 0.5 from step 0 + mem_read_layers [4, 8].",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, mqar_xchunk_start_frac=0.0, mem_read_layers=[4, 8], **X)
emit("screen_p25i_6L_chunk512_x_read", P, "+ chunk512/T1024 + xchunk 0.5 from step 0 + mem_read_layers [3]; B8xGA2.",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, mqar_xchunk_start_frac=0.0, mem_read_layers=[3], **X)
# 10-05 10:20: every run that forces memory use from the start costs 10-40% ppl (attention never matures). LATE MEMORY
# PHASE: resume r21b's final ckpt (37.59, 200M tok; copied into the run dir by temp/queue_5090_r25fg.sh) and train 40M more
# tokens with chunks + xchunk MQAR (the gap dose is already on past 70%), with / without the late read.
LATE = dict(block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, mqar_xchunk_start_frac=0.0, max_tokens=240_000_000)
emit("r25f_late_chunk_read_150m", "r21b_x1_150m", "LATE PHASE from r21b @200M: +40M tok chunk512/T1024 + xchunk + "
     "mem_read_layers [4, 8].", mem_read_layers=[4, 8], **LATE, **X)
emit("r25g_late_chunk_150m", "r21b_x1_150m", "LATE PHASE from r21b @200M: +40M tok chunk512/T1024 + xchunk (no late read).",
     **LATE, **X)
# 10-05 11:10: r25f = memory used at 12L with NO ppl cost (chunk1 38.34 vs r21b 38.84) but cross-chunk recall at chance
# -> stronger late dose: 80M tok, every gap batch cross-chunk, gap weight 0.5.
emit("r25h_late_read_xstrong_150m", "r21b_x1_150m", "LATE PHASE from r21b @200M: +80M tok chunk512/T1024 + late read "
     "[4, 8] + xchunk frac 1.0, gap weight 0.5.", mem_read_layers=[4, 8], mqar_gap_weight=0.5,
     **dict(LATE, max_tokens=280_000_000), **dict(X, mqar_xchunk_frac=1.0))
# 10-05 13:00: p25e (6L, 2nd memory at depth 3) = cross-chunk haystack N8 ~89% / N32 ~80% (mem off chance), +10% ppl.
# r25h (late phase, xchunk 1.0, w 0.5) = cross-chunk N8 59-79% / N32 ~30% at 12L, chunk1 +4%, in-window haystack fell
# (no in-chunk gap batches). Next: balanced late phase (r25i) and 12L from scratch with a 2nd memory layer (r25j).
emit("r25i_late_read_xbal_150m", "r21b_x1_150m", "LATE PHASE from r21b @200M: +80M tok chunk512/T1024 + late read "
     "[4, 8] + xchunk frac 0.5, gap weight 0.3.", mem_read_layers=[4, 8], mqar_gap_weight=0.3,
     **dict(LATE, max_tokens=280_000_000), **X)
emit("r25j_chunk_H6_150m", "r21b_x1_150m", "+ chunk512/T1024 + xchunk 0.5 + 2nd memory layer at depth 6 (H0 + H6).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, attention_layer_positions=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
     **X)
emit("screen_p25l_6L_chunk512_x_H3_nl", P, "p25e (H0 + H3) + NL-fact aux; B8xGA2.", block_size=1024, attn_chunk=512,
     mqar_xchunk_batch=4, attention_layer_positions=[1, 2, 4, 5], **X, **NL)
# 10-05 14:05: r25i (xchunk 0.5, w 0.3) = no ppl cost, memory used, cross-chunk at chance; r25h (1.0, w 0.5) = cross-chunk
# 59-79% but in-window haystack fell -> r25k in between: xchunk 0.75, w 0.5.
emit("r25k_late_read_x075_150m", "r21b_x1_150m", "LATE PHASE from r21b @200M: +80M tok chunk512/T1024 + late read "
     "[4, 8] + xchunk frac 0.75, gap weight 0.5.", mem_read_layers=[4, 8], mqar_gap_weight=0.5,
     **dict(LATE, max_tokens=280_000_000), **dict(X, mqar_xchunk_frac=0.75))
# 10-05 17:15: r25j (12L H0+H6 chunked from scratch) = cross-chunk haystack 100% (N8+N32) but chunk1 47.28 (+22%).
# r25l = LATE INSERTION: r21b ckpt + a NEW memory block at depth 6 (13L, all 11 attention layers kept; ckpt remapped by
# temp/mk_insert_ckpt.py; spk_layerscale 0.01 so the new block starts near-silent, loaded H0 gets ls_gamma 1) + 80M tok
# chunked with xchunk 0.5 at the default gap weight (r25j learned with that dose).
emit("r25l_late_insH6_150m", "r21b_x1_150m", "LATE PHASE from r21b @200M: insert memory block at depth 6 (13L) + "
     "chunk512/T1024 + xchunk 0.5, 80M tok.", n_layers=13, insert_blocks=[6], spk_layerscale=0.01,
     attention_layer_positions=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
     **dict(LATE, max_tokens=280_000_000, mqar_xchunk_batch=4), **X,
     # 10-05 20:55: 33.5 GB at B16 (WDDM spill, 3k tok/s) -> B8 x GA2, per-micro aux halved (same samples/step)
     batch_size=8, grad_accum=2, mqar_aug_batch=16, mqar_batch=4)
# 10-05 21:05 (5080 free): is p25e's +10% ppl from losing an attention layer, and is H0 needed once H3 exists?
emit("screen_p25m_7L_chunk512_x_H0H3", P, "7L: H0 + H3 INSERTED (all 5 attention layers kept) + chunk + xchunk; B8xGA2.",
     n_layers=7, block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, attention_layer_positions=[1, 2, 4, 5, 6], **X)
emit("screen_p25n_6L_chunk512_x_H3only", P, "6L: memory ONLY at depth 3 (layer 0 = attention) + chunk + xchunk; B8xGA2.",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, attention_layer_positions=[0, 1, 2, 4, 5], **X)
# 10-05 22:55: r25l chunk1 36.94 but the inserted H6 stayed silent (Δ 0, cross-chunk chance) -> r25m: full-scale insert
# (spk_layerscale 1.0) + strong dose (xchunk 1.0, gap weight 0.5).
emit("r25m_late_insH6_strong_150m", "r21b_x1_150m", "LATE PHASE from r21b: insert H6 at layerscale 1.0 (13L) + chunk + "
     "xchunk 1.0, gap w 0.5, 80M tok; B8xGA2.", n_layers=13, insert_blocks=[6], spk_layerscale=1.0, mqar_gap_weight=0.5,
     attention_layer_positions=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
     **dict(LATE, max_tokens=280_000_000, mqar_xchunk_batch=4), **dict(X, mqar_xchunk_frac=1.0),
     batch_size=8, grad_accum=2, mqar_aug_batch=16, mqar_batch=4)
# 10-06 00:50: p25m (7L, H3 ADDED, all attention kept) = +2.8% ppl, cross-chunk ~90/83% -> 12L analog from scratch.
emit("r25n_13L_H0H6add_150m", "r21b_x1_150m", "13L from scratch: H0 + H6 ADDED (all 11 attention layers kept) + chunk512/"
     "T1024 + xchunk 0.5 (late gap dose); B8xGA2.", n_layers=13, attention_layer_positions=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=4, batch_size=8, grad_accum=2, mqar_aug_batch=16, mqar_batch=4, **X)
# 10-06 03:05: p25n (H3 only) = cross-chunk 94/85% (best), +8.5% ppl -> p25o: H3 only, ADDED (6 attention layers kept).
emit("screen_p25o_7L_chunk512_x_H3add", P, "7L: attention at 0,1,2,4,5,6 + memory ONLY at depth 3 (added) + chunk + "
     "xchunk; B8xGA2.", n_layers=7, block_size=1024, attn_chunk=512, mqar_xchunk_batch=4,
     attention_layer_positions=[0, 1, 2, 4, 5, 6], **X)
# 10-06 05:00: p25o (7L, H3 only, added) = +0.8% ppl, cross-chunk ~91/81% -> 12L analog.
emit("r25o_13L_H6only_150m", "r21b_x1_150m", "13L from scratch: 12 attention layers + memory ONLY at depth 6 (added) + "
     "chunk512/T1024 + xchunk 0.5 (late gap dose); B8xGA2.", n_layers=13,
     attention_layer_positions=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], block_size=1024, attn_chunk=512,
     mqar_xchunk_batch=4, batch_size=8, grad_accum=2, mqar_aug_batch=16, mqar_batch=4, **X)
emit("screen_p25p_7L_H3add_nl", P, "p25o + NL-fact aux; B8xGA2.", n_layers=7, block_size=1024, attn_chunk=512,
     mqar_xchunk_batch=4, attention_layer_positions=[0, 1, 2, 4, 5, 6], **X, **NL)
emit("screen_p25q_7L_H3add_T2048", P, "p25o at T 2048 (3 chunk boundaries); B4xGA2.", n_layers=7, block_size=2048,
     attn_chunk=512, mqar_xchunk_batch=4, attention_layer_positions=[0, 1, 2, 4, 5, 6], **X)
# 10-06 05:15: r25n (13L H0+H6 added) = cross-chunk 99/93% but chunk1 +23%; chunking itself costs at 12L (r25a +11%).
# Suspect: half as many independent windows per step (16x1024 vs 32x512). p25r = p25o at T512 / chunk 256 (32 windows).
emit("screen_p25r_7L_H3add_T512c256", P, "p25o at T 512, chunk 256 (same window count as r23b); B16xGA2.", n_layers=7,
     block_size=512, attn_chunk=256, mqar_xchunk_batch=4, attention_layer_positions=[0, 1, 2, 4, 5, 6], **X)
# 10-06 08:30: r25o (13L, 12 attn + H6 only) = BEST 12L: chunk1 41.74 (+7.5%), cross-chunk ~100/84-100%, in-window 100%.
emit("r25q_13L_H6only_nl_150m", "r21b_x1_150m", "r25o + NL-fact aux (every 2nd step, w 0.15); B8xGA2.", n_layers=13,
     attention_layer_positions=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], block_size=1024, attn_chunk=512,
     mqar_xchunk_batch=4, batch_size=8, grad_accum=2, mqar_aug_batch=16, mqar_batch=4, **X, **NL)
# 10-06 09:30 ENERGY PLAN (docs/energy_plan_2026_10_06.md): way (1)+(3) = spike-code every linear input, as a late
# CONVERSION phase from the trained ckpt (seeded + calibrated by temp/mk_spike_seed.py).
SPK = dict(spike_code=["qkv", "o", "up", "down", "head", "mem"], spike_levels=4, spike_levels_hidden=8)
emit("screen_p25s_7L_spk_conv", "screen_p25o_7L_chunk512_x_H3add", "CONVERSION: p25o ckpt + 20M tok with spike_code "
     "on every linear input (L4 signed, hidden L8).", max_tokens=120_000_000, **SPK)
emit("r25s_13L_spk_conv_150m", "r25o_13L_H6only_150m", "CONVERSION: r25o ckpt + 40M tok with spike_code on every linear "
     "input (L4 signed, hidden L8).", max_tokens=240_000_000, **SPK)
# 10-06 10:10: r25s = 5.56x energy (from 1.39x), memory intact, chunk1 +7.8% -> sweep levels / thresholds / data.
emit("r25t_13L_spk_L8_80M_150m", "r25o_13L_H6only_150m", "CONVERSION: r25o + 80M tok, spike levels 8 (hidden 16).",
     max_tokens=280_000_000, **dict(SPK, spike_levels=8, spike_levels_hidden=16))
PO = "screen_p25o_7L_chunk512_x_H3add"
emit("screen_p25s2_7L_spk_L8", PO, "CONVERSION: p25o + 20M tok, levels 8 (hidden 16).", max_tokens=120_000_000,
     **dict(SPK, spike_levels=8, spike_levels_hidden=16))
emit("screen_p25s3_7L_spk_sc2", PO, "CONVERSION: p25o + 20M tok, L4, thresholds calibrated at 2x (fewer spikes).",
     max_tokens=120_000_000, **SPK)
emit("screen_p25s4_7L_spk_tern", PO, "CONVERSION: p25o + 20M tok, TERNARY spikes (L1 signed, hidden L2).",
     max_tokens=120_000_000, **dict(SPK, spike_levels=1, spike_levels_hidden=2))
emit("screen_p25t_7L_spk_scratch", PO, "p25o with spike_code from STEP 0 (way 1 without conversion), L4.", **SPK)
# 10-06 12:15 review: r25t (L8, 80M conv) = NO ppl cost (41.67 vs 41.74), 5.0x. Real-text recall beyond the window fails
# (xcopy 30 vs 28 mem-off) -> xcopy augmentation. Energy next: KD + 1.5x thresholds.
emit("r25v_13L_H6_xcopy_150m", "r25o_13L_H6only_150m", "r25o recipe + xcopy_frac 0.25 (cross-chunk copy augmentation), "
     "from scratch.", xcopy_frac=0.25)
emit("screen_p25u_7L_spk_kd", PO, "CONVERSION: p25o + 40M tok, L8, thresholds 1.5x (SCALE env), self-KD from p25o "
     "(alpha 0.5, every 4th position).", max_tokens=140_000_000, kd_self_ckpt=
     "/home/glenn/projects/neuro/models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt",
     kd_self_alpha=0.5, kd_self_stride=4, **dict(SPK, spike_levels=8, spike_levels_hidden=16))
emit("screen_p25v_7L_xcopy", PO, "p25o + xcopy_frac 0.25, from scratch.", xcopy_frac=0.25)
emit("r25c_chunk_x1_nl_150m", "r21b_x1_150m", "r25a + NL-fact aux (every 2nd step, w 0.15).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, **X, **NL)
