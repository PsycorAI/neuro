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
    if name in ("screen_p25g_6L_chunk512_x_fix", "screen_p25i_6L_chunk512_x_read", "screen_p25b_6L_chunk512", "screen_p25e_6L_chunk512_x_H3",
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
emit("r25c_chunk_x1_nl_150m", "r21b_x1_150m", "r25a + NL-fact aux (every 2nd step, w 0.15).",
     block_size=1024, attn_chunk=512, mqar_xchunk_batch=8, **X, **NL)
