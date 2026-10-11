"""mk_p26_configs.py -- 2026-10-06 (Neuro 52). p26 = fixes for real-text recall beyond the window
(docs/next_experiments_2026_10_06.md). E2 diagnostic: the memory's write gate opens only on the range-coded MQAR block
(0.83-0.91) and sits at beta_floor (0.10) on all text -> teach it to write ordinary-looking tokens.
Control = screen_p25o_7L_chunk512_x_H3add (7L, 6 attention + memory at depth 3, chunk512/T1024, xchunk 0.5)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"


def emit(name, src, note, **upd):
    c = yaml.safe_load(open(R + src + ".yaml"))
    c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_p26_configs.py (2026-10-06) from {src}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("wrote", name)


P = "screen_p25o_7L_chunk512_x_H3add"
emit("screen_p26e_7L_xscatter", P, "p25o + mqar_xchunk_scatter 1.0: every cross-chunk MQAR batch scatters whole-vocab "
     "kv pairs through chunk A's real text (no token-range / position cue).", mqar_xchunk_scatter=1.0)
# 10-06 22:45 p26e: write gate opens on unusual tokens (0.30 vs 0.10), real-text copy gain near the boundary ~3x (+8 vs
# +2.5 pts) BUT val_ppl 77.0 vs 66.0 (+17%), block haystack 41-73% vs ~91%, multi-chunk retention collapsed -> dose down.
emit("screen_p26g_7L_xscatter25", P, "p25o + mqar_xchunk_scatter 0.25 (dose test of p26e).", mqar_xchunk_scatter=0.25)
emit("screen_p26h_7L_xphrase25", P, "p25o + mqar_xchunk_scatter 0.25 in PHRASE mode: 2-token keys of ordinary words, "
     "1-token word values, scattered in real text (vs p26g whole-vocab single tokens).", mqar_xchunk_scatter=0.25,
     mqar_scatter_mode="phrase")

# 10-06 23:20 E4 (temp/energy_eval_v2.py): with weight-read energy the advantage shrinks -- r25w 7.5x compute-only ->
# 3.95x with on-chip SRAM reads -> 2.26x DRAM-streamed (r25t 5.0 -> 3.1 -> 1.9), because ~45-52% of linear inputs are
# still NONZERO and each nonzero input costs a weight read. The honest lever = the nonzero fraction -> higher thresholds.
# p26k / p26l: p25u conversion recipe (L8 + self-KD, 40M tok) + q/v coding at threshold scale 2.0 / 3.0 (SCALE env).
for _n, _s in (("k", "20"), ("l", "30")):
    emit(f"screen_p26{_n}_7L_spk_kd_qv_sc{_s}", "screen_p25u_7L_spk_kd", f"CONVERSION: p25u recipe + q/v coding at "
         f"threshold scale {_s[0]}.{_s[1]} (set by the queue) -- fewer nonzero inputs for honest (weight-read) energy.",
         spike_code=["qkv", "o", "up", "down", "head", "mem", "q", "v"])

# 10-06 23:10 Glenn's idea: accept the memory-training ppl cost, recover it at the end. POLISH phases from the p26e ckpt
# (seeded by temp/mk_polish_seed.py): +25M tok, scatter dose 1.0 -> 0.25 (rehearsal, keeps the skill).
#   p26m = anneal + low-dose rehearsal only;  p26n = + KD from p25o (same layout, val_ppl 66 vs p26e 77).
E = "screen_p26e_7L_xscatter"
emit("screen_p26m_7L_polish", E, "POLISH: p26e ckpt + 25M tok, scatter 0.25 rehearsal (no KD).", max_tokens=125_000_000,
     mqar_xchunk_scatter=0.25)
emit("screen_p26n_7L_polish_kd", E, "POLISH: p26e ckpt + 25M tok, scatter 0.25 rehearsal + KD from p25o (alpha 0.5, every "
     "4th position).", max_tokens=125_000_000, mqar_xchunk_scatter=0.25, kd_self_ckpt=
     "/home/glenn/projects/neuro/models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt", kd_self_alpha=0.5,
     kd_self_stride=4)

# 10-07 02:15 p26h (phrase 0.25) = 72.19, same +9% as p26g (72.76) -> rare-token targets are NOT the cause. p25o is a
# single seed (siblings 65.6-68.0): p26s = p25o seed 1 calibrates the 7L noise for every p26 comparison.
emit("screen_p26s_7L_p25o_seed1", P, "p25o, seed 1 (noise control).", seed=1)

# 10-07 05:30 polish results: p26m 75.05, p26n (+KD) 74.72 vs p26e 77.03 / p25o 66.04 -> only ~2-3 of 11 pts back, KD
# ~nothing; memory gains kept. Both ran at the cosine TAIL (LR ~10% of peak) -> not a fair test of "recover at the end".
# p26q / p26r: same polish with a REWARMED LR: lr_schedule wsd with the decay starting at the resume step (decay frac
# (7629-6104)/(7629-152) = 0.204 -> 0.21), peak lr 0.3x (lr 1.8e-4, muon_lr 1.5e-3), linear decay to 0.1x over 25M.
RW = dict(max_tokens=125_000_000, mqar_xchunk_scatter=0.25, lr_schedule="wsd", wsd_decay_frac=0.21, lr=0.00018,
          muon_lr=0.0015)
emit("screen_p26q_7L_polish_rw", E, "POLISH: p26e ckpt + 25M tok, rewarmed LR 0.3x then linear decay, scatter 0.25.", **RW)
emit("screen_p26r_7L_polish_rw_kd", E, "POLISH: p26q + KD from p25o (alpha 0.5, every 4th position).", **RW,
     kd_self_ckpt="/home/glenn/projects/neuro/models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt",
     kd_self_alpha=0.5, kd_self_stride=4)

# 10-07 07:20 energy curve at 7L / 40M conversion (chunk1 vs dense p25o 69.0; SRAM = with weight reads):
#   1.5x (p25u) +10%, ~4x SRAM | 2x (p26k) 85.75 +24%, 10.3x compute / 5.1x SRAM | 3x (p26l) 110.2 +60%, 14.7x / 7.4x.
# 10-07 08:40 rewarmed polish: p26q 76.44, p26r (+KD) 75.50 -> no better than the cosine-tail polish (75.05 / 74.72):
# with the scatter aux still on (0.25) the ppl cost does not come back. p26v: rewarmed polish with scatter OFF (plain
# block xchunk as in p25o) -> is the cost the LIVE aux or the learned weights, and how much memory gain survives?
emit("screen_p26v_7L_polish_rw_off", E, "POLISH: p26e ckpt + 25M tok, rewarmed LR (as p26q), scatter OFF.",
     **dict(RW, mqar_xchunk_scatter=0.0))
# p26u: does a 3x LONGER conversion (120M tok) shrink the 2x-threshold cost? (sizes the v4 cloud conversion phase)
emit("screen_p26u_7L_spk_kd_qv_sc20_120M", "screen_p26k_7L_spk_kd_qv_sc20", "CONVERSION: p26k recipe (2x thresholds + q/v "
     "+ self-KD) for 120M tok instead of 40M.", max_tokens=220_000_000)

# ENERGY weakness (honest = weight reads, set by the nonzero-input fraction): raise thresholds only where the energy is
# (r25x: head 34 uJ, up 29, qkv 20.5 of 122). p26w = p26k recipe but SITE_SCALE head=2,up=2 (rest 1.0), 40M tok.
emit("screen_p26w_7L_spk_kd_qv_site", "screen_p26k_7L_spk_kd_qv_sc20", "CONVERSION: p26k recipe at 1.0x thresholds except "
     "head / MLP input 2x (SITE_SCALE, set by the queue).")

# 10-07 11:20 p26u (2x, 120M) = chunk1 72.27 (+4.7% vs dense 69.0; p26k 40M +24%) BUT energy 8.52x / 4.35x SRAM (p26k
# 10.3x / 5.1x): the learned thresholds drift down. p26x = p26u with spike_theta_frozen (energy pinned at the 2x level).
emit("screen_p26x_7L_spk_kd_qv_sc20_120M_frz", "screen_p26u_7L_spk_kd_qv_sc20_120M", "CONVERSION: p26u (2x, 120M tok) with "
     "thresholds FROZEN at calibration.", spike_theta_frozen=True)

# 10-07 13:20 REVIEW idea (lit: SWAX 2509.24552 trains with a RANDOM window, 128 w.p. 0.5 else 2048, fixed for the last
# 10% -> short-context quality equal/better AND far better long recall). Our analogue: random attention CHUNK size
# (attn_chunk_short, new flag): with 128-token chunks ordinary text crosses boundaries all the time, so the LM loss itself
# trains the memory to carry REAL text -- the signal E2 showed missing -- with no synthetic aux and its ppl cost.
SC = dict(attn_chunk_short=128, attn_chunk_short_p=0.5, attn_chunk_anneal_frac=0.1)
emit("screen_p26y_7L_chunk_rand128", P, "p25o + stochastic chunk (128 w.p. 0.5, else 512; fixed 512 for the last 10%).", **SC)
emit("r26f_13L_H6_chunk1024_rand128_150m", "r26c_13L_H6_chunk1024_150m", "r26c (RC layout) + stochastic chunk (128 w.p. 0.5, "
     "else 1024; fixed 1024 for the last 10%).", **SC)

# ---- 5090 (12L), queued after Neuro 52's r25z (way 2) ----
# r25t (L8, 80M, scale 1.0, no KD, no q/v) = 5.0x at no ppl cost; r25w (+ q/v + self-KD + 1.5x thresholds) = 7.5x at
# +6.4%. r25x isolates the thresholds: r25w recipe at calibration scale 1.0 (SCALE env of temp/queue_spk_conv.sh).
emit("r25x_13L_spk_kd_sc10_150m", "r25w_13L_spk_kd_sc15_150m", "CONVERSION: r25w recipe (L8 + q/v + self-KD, 80M tok) "
     "at threshold scale 1.0 (set by the queue).")
# Demo context: attention chunks of 1024 (T2048, 2 chunks) instead of 512 -- longer in-window context for chat, memory
# beyond. Same tokens/step (B4xGA2), xchunk batches padded to 2C = 2048.
_r = yaml.safe_load(open(R + "r25o_13L_H6only_150m.yaml"))
emit("r26c_13L_H6_chunk1024_150m", "r25o_13L_H6only_150m", "r25o with attn_chunk 1024 / T2048 (B4xGA2, same tokens/step).",
     block_size=2048, attn_chunk=1024, batch_size=4, eval_batch=max(1, _r["eval_batch"] // 2))
# r25o is a single seed (the planned v4 layout): r26s = seed 1, to confirm chunk1 cost + cross-chunk recall before the freeze.
emit("r26s_13L_H6_seed1_150m", "r25o_13L_H6only_150m", "r25o, seed 1 (robustness of the v4 layout).", seed=1)
# 10-07 09:15: r26c (chunk 1024) is the recommended v4 layout -> r26s replaced (stopped after 5 min) by r26t = r26c seed 1.
emit("r26t_13L_H6_chunk1024_seed1_150m", "r26c_13L_H6_chunk1024_150m", "r26c, seed 1 (robustness of the v4 layout).", seed=1)
# ---- RELEASE CANDIDATE (10-07, Glenn: RC with good recall, then strengthen ppl / energy / throughput) ----
# RC = r26c layout (12 attn + memory at 6, chunk 1024 / T2048, xchunk 0.5 from 30%) + r25x conversion (L8 + q/v + self-KD,
# 80M tok, thresholds 1.0x). r26d = that conversion on the r26c ckpt (confirms it is free at chunk 1024).
KDX = dict(kd_self_alpha=0.5, kd_self_stride=4, spike_code=["qkv", "o", "up", "down", "head", "mem", "q", "v"],
           spike_levels=8, spike_levels_hidden=16, mqar_aug_batch=8, max_tokens=280_000_000)
emit("r26d_13L_RC_conv_150m", "r26c_13L_H6_chunk1024_150m", "RC CONVERSION: r26c ckpt + 80M tok, r25x recipe (L8 + q/v + "
     "self-KD from r26c, thresholds 1.0x).", kd_self_ckpt=
     "/home/glenn/projects/neuro/models/checkpoints/r26c_13L_H6_chunk1024_150m/step_0012208.pt", **KDX)
# PPL weakness: the xchunk aux bumps val_ppl when it starts (r25o 80.8 -> 83.8, r25z far worse). r26e = r26c with half
# the cross-chunk dose (0.25) starting later (0.5) -- less aux cost; does cross-chunk recall still reach ~100%?
emit("r26e_13L_H6_chunk1024_xlate_150m", "r26c_13L_H6_chunk1024_150m", "r26c with mqar_xchunk_frac 0.25 from 50% "
     "(was 0.5 from 30%).", mqar_xchunk_frac=0.25, mqar_xchunk_start_frac=0.5)
# ---- HOLA hippocampus (10-07 15:40, Glenn: 1st choice = in v4 from the start). hola_w = exact cache size (paper: 64),
# boundaries = attention chunks, score = |u_t| (committed delta-rule correction), zero-init lambda.
emit("screen_p26z_7L_hola64", P, "p25o + HOLA exact cache (64) in the memory block.", hola_w=64)
emit("screen_p26za_7L_hola64_rand128", P, "p25o + HOLA (64) + stochastic chunk (128 w.p. 0.5): many cache boundaries.",
     hola_w=64, **SC)
emit("r26h_13L_RC_hola64_150m", "r26c_13L_H6_chunk1024_150m", "RC layout (r26c) + HOLA exact cache (64).", hola_w=64)

# PPL weakness, review idea: the conversion phase already pays for KD -- use a REAL teacher (Llama-3.2-1B base, src/kd.py,
# LUT-projected to our 32k vocab) instead of self-KD, so the converted RC may end BELOW the dense ppl. r26g vs r26d.
_g = {k: v for k, v in KDX.items() if not k.startswith("kd_self")}
emit("r26g_13L_RC_conv_llamakd_150m", "r26c_13L_H6_chunk1024_150m", "RC CONVERSION with external KD: r26c ckpt + 80M tok, "
     "r25x spike recipe, teacher Llama-3.2-1B (alpha 0.5) instead of self-KD.", kd_enabled=True,
     kd_teacher="meta-llama/Llama-3.2-1B", kd_alpha=0.5, kd_temperature=1.0, **_g)
# 10-08 03:45 r26h (RC + HOLA 64) = WIN: chunk1 39.30 (r26c/r26t 40.27/41.33), cross-chunk N32 ~100 at every offset,
# retention N32 97% at chunk 8 (r26t 69%), real-text copy gain ~2x r26c, same tok/s -> RC2 = RC + HOLA.
# r26i = RC2 + stochastic chunk (the full memory combination); r26j = RC2 conversion (r25x recipe, self-KD from r26h).
emit("r26i_13L_RC2_hola_rand128_150m", "r26h_13L_RC_hola64_150m", "RC2 (RC + HOLA 64) + stochastic chunk (128 w.p. 0.5, "
     "else 1024; fixed 1024 for the last 10%).", **SC)
emit("r26j_13L_RC2_conv_150m", "r26h_13L_RC_hola64_150m", "RC2 CONVERSION: r26h ckpt + 80M tok, r25x recipe (L8 + q/v + "
     "self-KD from r26h, thresholds 1.0x).", kd_self_ckpt=
     "/home/glenn/projects/neuro/models/checkpoints/r26h_13L_RC_hola64_150m/step_0012208.pt", **KDX)
# 10-08 12:55: r26k = RC2 (r26h) seed 1 -- robustness of the HOLA win (ppl gain ~ seed spread). For the 16 GB 5080:
# B2xGA4 (same tokens/step), WITHOUT expandable segments (both 'device not ready' crashes had them on).
emit("r26k_13L_RC2_hola_seed1_150m", "r26h_13L_RC_hola64_150m", "RC2 (r26h) seed 1; B2xGA4 for the 5080.", seed=1,
     batch_size=2, grad_accum=4, eval_batch=1, mqar_aug_batch=8, mqar_batch=2, mqar_xchunk_batch=2)
# 10-08 13:15 overnight (Glenn: 5080 free all night). All from RC2 (r26h):
# r26g2 = RC2 conversion with Llama-3.2-1B KD, B2xGA4 (r26g spilled at B4xGA2: teacher logits 128k-vocab);
# r26l  = RC2 conversion, thresholds 2x FROZEN, 120M tok (energy lever; 7L p26x frozen kept 4.8x SRAM vs 4.35x);
# r26m  = RC2 with a 4x larger HOLA cache (256) -- more real-text facts beyond the window? (5080, B2xGA4)
H = "/home/glenn/projects/neuro/models/checkpoints/r26h_13L_RC_hola64_150m/step_0012208.pt"
B24 = dict(batch_size=2, grad_accum=4, eval_batch=1, mqar_batch=2, mqar_xchunk_batch=2)
emit("r26g2_13L_RC2_conv_llamakd_150m", "r26h_13L_RC_hola64_150m", "RC2 CONVERSION with Llama-3.2-1B KD (alpha 0.5), "
     "B2xGA4.", kd_enabled=True, kd_teacher="meta-llama/Llama-3.2-1B", kd_alpha=0.5, kd_temperature=1.0,
     **dict(_g, mqar_aug_batch=4), **B24)
emit("r26l_13L_RC2_conv_sc20frz_120M_150m", "r26h_13L_RC_hola64_150m", "RC2 CONVERSION: r25x recipe at thresholds 2x "
     "FROZEN, 120M tok.", kd_self_ckpt=H, spike_theta_frozen=True, **dict(KDX, max_tokens=320_000_000))
emit("r26m_13L_RC2_hola256_150m", "r26h_13L_RC_hola64_150m", "RC2 with HOLA cache 256 (was 64); B2xGA4 for the 5080.",
     hola_w=256, mqar_aug_batch=8, **B24)
# 10-08 19:45 r26e (RC + xchunk 0.25 from 50%) = chunk1 39.01 (r26c 40.27), val 39.66, memory slightly weaker -> r26n =
# RC2 (HOLA restores memory) + the same dose-down: best ppl + memory?
emit("r26n_13L_RC2_xlate_150m", "r26h_13L_RC_hola64_150m", "RC2 + mqar_xchunk_frac 0.25 from 50% (r26e dose-down).",
     mqar_xchunk_frac=0.25, mqar_xchunk_start_frac=0.5)
# 10-09 07:45 r26n (RC2 + xchunk 0.25 from 50%) = chunk1 38.09 (r26h 39.30), retention 100/100 at chunk 8 -> RC3.
# r26o = RC3 conversion with Llama-3.2-1B KD (the r26g2 recipe that gave 37.48 on RC2), B2xGA4.
emit("r26o_13L_RC3_conv_llamakd_150m", "r26n_13L_RC2_xlate_150m", "RC3 CONVERSION with Llama-3.2-1B KD (alpha 0.5), B2xGA4.",
     kd_enabled=True, kd_teacher="meta-llama/Llama-3.2-1B", kd_alpha=0.5, kd_temperature=1.0,
     **dict(_g, mqar_aug_batch=4), **B24)
# 10-09 18:00 (Neuro 54; Glenn: test the long-run guard before the v4 pretrain). r26p = RC3 dense (r26n) +
# head_weight_decay 1.0 (v3 divergence fix, in no r2x config, not resume-compatible). No-harm test only: 150M cannot
# show the late divergence itself. Judge vs r26n chunk1 38.09 / val 38.72 (seed noise ~2.6%: r26c 40.27 vs 41.33).
emit("r26p_13L_RC3_headwd_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense (r26n) + head_weight_decay 1.0 (no-harm test "
     "for v4_cloud).", head_weight_decay=1.0)
# 10-09 20:00 r26k (RC2 seed 1) 42.69 / r26m (HOLA 256) 41.35 vs r26h 39.30 -- both killed + resumed 3-4x at B2xGA4, and
# in both val_ppl goes FLAT after the last resume at step 10000 (r26h still gains 9% there). Is a late resume lossy?
# r26q = r26n EXACTLY (seed 0, B4xGA2) with ONE planned SIGTERM + resume right after step 10000 (temp/kill_at_step.sh).
# Lossless resume -> ~38.09 (r26n val 42.21 / 40.12 / 38.72 at steps 10k / 11k / 12k). Decides the interruptible cloud.
emit("r26q_13L_RC3_resume10k_150m", "r26n_13L_RC2_xlate_150m", "RESUME CONTROL: r26n with one planned SIGTERM + resume "
     "after step 10000.")
# 10-09 23:00 r26k (seed 1) was confounded by kills + B2xGA4 -> r26r = RC3 dense (r26n) seed 1, uninterrupted, B4xGA2:
# the clean seed-robustness number for the v4 recipe (r26n seed 0 = 38.09).
emit("r26r_13L_RC3_seed1_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense (r26n), seed 1 (clean seed test).", seed=1)
# 10-10 02:10 cloud go/no-go timing: RC3 starts the cross-chunk aux at 50% = ~1.5 days into the H200 run before the
# memory can be judged. Does the SAME dose (0.25) from 20% cost ppl? (r26e/r26n only tested dose + start together.)
# r26s = 12L on the 5090 (decides); p26za / p26zb = 7L pair on the 5080 (start 50% vs 20%, 100M tok, early signal only:
# memory ideas are judged at 12L). 7L pair = configs/v4_reh.yaml (the rehearsal's 7L RC3 layout) at 100M.
# 10-10 04:45 r26q = chunk1 39.56 (+3.9%), val 43.00 -> 40.95 -> 40.29 after the resume (r26n 42.21 -> 40.12 -> 38.72) and
# its weight norms shrink 6x slower after the resume -> ROOT CAUSE: train_gpu read the schedule's base LRs AFTER loading
# the optimizer state (already-scheduled lr) -> each resume multiplied the LR by lr_scale(save step) (x0.17 at 82%).
# FIXED in train_gpu.py (temp/test_resume_lr.py). r26t = r26q again with the fix: must match r26n / r26q's pre-stop curve.
emit("r26t_13L_RC3_resume10k_fix_150m", "r26n_13L_RC2_xlate_150m", "RESUME CONTROL after the resume-LR fix: r26n with "
     "one planned SIGTERM + resume after step 10000.")
emit("r26s_13L_RC3_x20_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense (r26n) with the xchunk aux from 20% (was 50%).",
     mqar_xchunk_start_frac=0.2)
_7 = dict(max_tokens=100_000_000, eval_every=1000, ckpt_every=1000, keep_last=2, watchdog_valppl_only=False)
emit("screen_p26za_7L_RC3_x50", "v4_reh", "7L RC3 (v4_reh layout), xchunk 0.25 from 50%, 100M tok (control).", **_7)
emit("screen_p26zb_7L_RC3_x20", "v4_reh", "7L RC3 (v4_reh layout), xchunk 0.25 from 20%, 100M tok.", **_7,
     mqar_xchunk_start_frac=0.2)
# 10-10 08:30 7L pair: x50 chunk1 55.99 (chunks2+ memory gain +8.33) | x20 55.39 (+6.52) -> no ppl cost of the early
# start at 7L (single seed; same-seed noise ~2%). Seed 1 of both makes it a 2-seed signal while the 5080 is free.
# (xchunk_haystack / nl_fact evals OOM on the 16 GB 5080 for these dense 7L T2048 ckpts at B32: chunk + xcopy only.)
emit("screen_p26zc_7L_RC3_x50_s1", "v4_reh", "7L RC3, xchunk 0.25 from 50%, seed 1.", **_7, seed=1)
emit("screen_p26zd_7L_RC3_x20_s1", "v4_reh", "7L RC3, xchunk 0.25 from 20%, seed 1.", **_7, seed=1,
     mqar_xchunk_start_frac=0.2)
# 10-10 16:45 r26s = 38.21 and the 7L pair tied -- but the test was a NO-OP: xchunk batches are drawn only inside the
# gap-MQAR branch, which starts at mqar_gap_start_frac = 0.7 in every RC config, so xchunk_start 0.5 and 0.2 both mean
# "from 70%" (r26s / p26za-zd are same-seed replicates: chunk1 noise 38.09 / 38.27 / 38.21 = ~0.5%, 7L ~1%).
# r26v = the REAL early start: gap MQAR and xchunk both from 20% (ramp 0.1 as before). Costs step time too (gap
# batches run for 80% of the run instead of 30%). Judge vs 38.09-38.27 and memory 100%.
emit("r26v_13L_RC3_gap20_x20_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense with gap MQAR + xchunk aux both from 20% "
     "(RC3: both effectively from 70%).", mqar_gap_start_frac=0.2, mqar_xchunk_start_frac=0.2)
# 10-10 09:30 r26r (RC3 dense seed 1, clean) = 38.65 (r26n 38.09) -> RC3 dense is seed-robust. r26u = the r26o
# conversion (Llama-3.2-1B KD, B2xGA4) from r26r: 2nd seed of the 36.55 headline, and the first conversion that is
# not resumed (r26o's last 6% ran at ~0.1x LR after the outage resume).
emit("r26u_13L_RC3_conv_llamakd_s1_150m", "r26r_13L_RC3_seed1_150m", "RC3 CONVERSION with Llama-3.2-1B KD from r26r "
     "(seed 1), B2xGA4.", kd_enabled=True, kd_teacher="meta-llama/Llama-3.2-1B", kd_alpha=0.5, kd_temperature=1.0,
     **dict(_g, mqar_aug_batch=4), **B24)
# 10-10 18:00 (Neuro 55; Glenn: test a milder head weight decay). r26p (1.0) = 38.92 vs same-seed replicates 38.09 /
# 38.27 / 38.21 (noise ~0.5%) = a real ~2% cost. r26w / r26x = 0.3 / 0.1: does the cost shrink while the head norm
# stays bounded ([wnorm] head: r26p 66 vs r26n 214)? Judge chunk1 vs 38.09-38.27 and the head norm at the end.
emit("r26w_13L_RC3_headwd03_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense (r26n) + head_weight_decay 0.3.",
     head_weight_decay=0.3)
emit("r26x_13L_RC3_headwd01_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense (r26n) + head_weight_decay 0.1.",
     head_weight_decay=0.1)
# 10-10 23:40 (Neuro 55; Glenn: the 5080 is free all night + 10-11). 7L twins of the two open 12L questions, 100M tok.
# NOTE: v4_reh now carries head_weight_decay 1.0, but p26za-zd RAN WITHOUT it (no head_weight_decay line in their
# logs) -> they are the no-decay baseline: seed 0 55.99 / 55.39, seed 1 55.45 / 55.43 (chunk1). Set it explicitly here.
#  ze / zf / zh = head_weight_decay 1.0 / 0.3 / 0.1 (seed 0), zj = 1.0 seed 1: the cost curve at 7L.
#  zg / zi = gap MQAR + xchunk both from 20% (the r26v test), no decay, seeds 0 / 1.
_n = dict(_7, head_weight_decay=None)
emit("screen_p26ze_7L_RC3_hwd10", "v4_reh", "7L RC3 + head_weight_decay 1.0.", **dict(_7, head_weight_decay=1.0))
emit("screen_p26zf_7L_RC3_hwd03", "v4_reh", "7L RC3 + head_weight_decay 0.3.", **dict(_7, head_weight_decay=0.3))
emit("screen_p26zg_7L_RC3_gap20_x20", "v4_reh", "7L RC3, gap MQAR + xchunk both from 20%, no head decay.", **_n,
     mqar_gap_start_frac=0.2, mqar_xchunk_start_frac=0.2)
emit("screen_p26zh_7L_RC3_hwd01", "v4_reh", "7L RC3 + head_weight_decay 0.1.", **dict(_7, head_weight_decay=0.1))
emit("screen_p26zi_7L_RC3_gap20_x20_s1", "v4_reh", "7L RC3, gap MQAR + xchunk both from 20%, no head decay, seed 1.",
     **_n, seed=1, mqar_gap_start_frac=0.2, mqar_xchunk_start_frac=0.2)
emit("screen_p26zj_7L_RC3_hwd10_s1", "v4_reh", "7L RC3 + head_weight_decay 1.0, seed 1.",
     **dict(_7, head_weight_decay=1.0), seed=1)
# 10-11 00:30 (Neuro 55, overnight review). The conversion headline has no like-for-like control: r26o (36.55) = r26n
# + 80M tokens + Llama KD + spike codes, compared with r26n at 200M tokens and no teacher (38.09). r26y = r26o WITHOUT
# the spike codes (same +80M tokens, KD, B2xGA4, LR profile; seed = r26n's last ckpt without optimizer state,
# temp/mk_kd_control_seed.py). Conversion cost in ppl = r26o 36.55 - r26y. Queued last on the 5090 (after r26x).
emit("r26y_13L_RC3_kdonly_150m", "r26o_13L_RC3_conv_llamakd_150m", "CONTROL for r26o: +80M tok with Llama KD, NO spike codes.",
     spike_code=None)
# 10-11 13:30 (Neuro 55). The last two 5080 slots: p26zi (early start, seed 1) and p26zj (decay 1.0, seed 1) add little
# (early start is negative at 13L and 7L; 7L shows no decay cost at any value) -> full 7L TRAINING runs with the new
# memory flags instead, against the no-decay 7L band (seed 0: 55.99 / 55.39, seed 1: 55.45 / 55.43):
#   zk = mqar_separate_backward + mqar_aux_splits 2, seed 0     zl = mqar_separate_backward, seed 1
emit("screen_p26zk_7L_RC3_sepbw_s2", "v4_reh", "7L RC3, no head decay, mqar_separate_backward + 2 aux splits.",
     **_n, mqar_separate_backward=True, mqar_aux_splits=2)
emit("screen_p26zl_7L_RC3_sepbw_s1", "v4_reh", "7L RC3, no head decay, mqar_separate_backward, seed 1.",
     **_n, seed=1, mqar_separate_backward=True)
# 10-11 15:20 (Neuro 55). 1B profile on the 5090 (logs/speed1b_prof_dense.log): the NorMuon step (Newton-Schulz on 60
# matrices, ~40 TFLOP) is 354 ms of GPU time per optimizer step, whatever the batch -- 89% of the GPU kernel time at
# 2,048 tok/step, and an estimated third of an H200 step at 32,768 tok/step. Its cost per TOKEN halves if a step
# carries twice the tokens. r26z = r26n with B4 x GA4 = 32,768 tok/step (r26n: B4 x GA2 = 16,384), same 200M tokens ->
# half the optimizer steps (6,103). Aux doses per token unchanged (the aux runs per micro-step). Judge chunk1 against
# the 13L band 38.09-39.00: inside = halving the steps is harmless even in this step-starved 150M regime.
emit("r26z_13L_RC3_tok32k_150m", "r26n_13L_RC2_xlate_150m", "RC3 dense (r26n) at 32,768 tok/step (B4 x GA4): half the "
     "optimizer steps for the same 200M tokens.", grad_accum=4, eval_every=500, ckpt_every=500)
