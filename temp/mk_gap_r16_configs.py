"""mk_gap_r16_configs.py -- 2026-10-02 round 16 (Glenn OK 10-02): v4 candidate + leave-one-out ablations, 4 layers.
Plan: docs/v4_plan_review_2026_10_01.md 6.5. Flags: temp/parity_r16_flags.py (bit-identical when off), CPU smoke
temp/smoke_r16_cpu.py. Reference: r14d 4L [S, A, A, A] = 45.41 (no canon/ramp, old hash, GELU).
  c0 (s0, s1): r14b recipe at 4L [H, (A + DSN) x 3] + Engram high-bit hash + gated attention output + canon taps
               in attention blocks + H without output LayerNorm (linear atoms; init increment RMS 0.26) + ternary
               spikes (lif_levels 1) + top-k 16 attention (r15c: free) + DSN = SpikeGLU k 256/2752 with late k-ramp
               (dense until 60%, geometric to k by 90%)
  l1: c0 with the old low-bit Engram hash     l2: c0 without the attention output gate
  l3: c0 with GELU MLPs instead of DSN        l4: c0 with lif_levels 4
  l5: c0 with the H output LayerNorm kept     x1: c0 + late-only gap dose (gap MQAR 440 at w 0.15 from 70%)
  x2: c0 with DSN-LIF neurons (LIF gate, learnable leak, threshold + sparsity penalty instead of k-WTA)
  t4: fair 4L transformer (r15e + Engram high-bit hash + gated attention + canon; full attention, GELU, AdamW)"""
import re
D = "/home/glenn/projects/neuro/configs/"
def make(src, name, why, edits, append=""):
    s = open(D + src + ".yaml").read()
    s = re.sub(r"^(# .*\n)+", "", s, count=1)
    s = f"# {name} -- round 16 (2026-10-02). AUTO-GEN temp/mk_gap_r16_configs.py from {src}. Change: {why}\n" + s
    for pat, rep in edits + [(rf"^run_name: {src}$", f"run_name: {name}")]:
        s, n = re.subn(pat, rep, s, flags=re.M)
        assert n == 1, (name, pat, n)
    s = s.rstrip("\n") + "\n" + append
    open(D + name + ".yaml", "w").write(s)
    print("[OK]", name)

R14B = "screen_r14b_3Lattn2_canon_ramp10_s0_30m"
L4 = [(r"^n_layers: 3$", "n_layers: 4"), (r"^- 2$", "- 2\n- 3")]
LEV1 = [(r"^lif_levels: 4.*$", "lif_levels: 1")]
COMMON = ("engram_v2_hash: hi\n"
          "attn_out_gate: true\n"
          "attn_canon: 4\n"
          "spk_out_norm: false\n"
          "attn_topk: 16\n")
DSN = ("attn_ffn_spikeglu_k: 256\n"
       "attn_ffn_hidden: 2752\n")
RAMP = ("attn_ffn_spikeglu_ramp_start: 0.6\n"
        "attn_ffn_spikeglu_ramp_end: 0.9\n")
C0 = COMMON + DSN + RAMP
WHY0 = "4L [H, (A+DSN)x3] candidate (see header of temp/mk_gap_r16_configs.py)"

make(R14B, "screen_r16_c0_s0_30m", WHY0 + ".", L4 + LEV1, C0)
make(R14B, "screen_r16_c0_s1_30m", WHY0 + ", seed 1.", L4 + LEV1 + [(r"^seed: 0$", "seed: 1")], C0)
make(R14B, "screen_r16_l1_oldhash_s0_30m", "c0 with the old low-bit Engram hash.", L4 + LEV1,
     C0.replace("engram_v2_hash: hi\n", ""))
make(R14B, "screen_r16_l2_nogate_s0_30m", "c0 without the attention output gate.", L4 + LEV1,
     C0.replace("attn_out_gate: true\n", ""))
make(R14B, "screen_r16_l3_gelu_s0_30m", "c0 with GELU MLPs instead of DSN.", L4 + LEV1, COMMON)
make(R14B, "screen_r16_l4_lev4_s0_30m", "c0 with lif_levels 4 (multi-level spikes).", L4, C0)
make(R14B, "screen_r16_l5_outnorm_s0_30m", "c0 with the H output LayerNorm kept.", L4 + LEV1,
     C0.replace("spk_out_norm: false\n", ""))
make(R14B, "screen_r16_x1_lategap_s0_30m", "c0 + late-only haystack gap dose (gap MQAR 440 at w 0.15 from 70%).",
     L4 + LEV1, C0 + "mqar_gap_max: 440\nmqar_gap_start_frac: 0.7\nmqar_gap_ramp_frac: 0.1\n"
                     "mqar_gap_weight: 0.15\nmqar_batch: 16\n")
make(R14B, "screen_r16_x2_dsnlif_s0_30m", "c0 with DSN-LIF neurons (LIF gate, learnable leak, sparsity penalty).",
     L4 + LEV1, COMMON + DSN + "attn_ffn_spikeglu_lif: true\n")
make("screen_r15e_tfmr3_fair_s0_30m", "screen_r16_t4_tfmr4_fair_s0_30m",
     "fair 4L transformer: r15e + Engram high-bit hash + gated attention + canon (full attention, GELU, AdamW).",
     [(r"^n_layers: 3$", "n_layers: 4"), (r"^- 2\nattention_heads", "- 2\n- 3\nattention_heads")],
     "engram_v2_hash: hi\nattn_out_gate: true\nattn_canon: 4\n")
