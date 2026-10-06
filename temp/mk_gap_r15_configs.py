"""mk_gap_r15_configs.py -- 2026-10-01 round 15 (Glenn OK): interpretable/spiking diagnosis round.
Plan: docs/interpretable_spiking_plan_2026_10_01.md. Flags: temp/parity_r15_flags.py (bit-identical when off).
  r15a: r14b, attention layers WITHOUT their MLP          (D1: is r13a's gain attention or MLP? 57M params)
  r15b: r14b, GELU MLP -> SpikeGLU k-WTA 256/2752          (S1: spiking MLP, +0.7% params)
  r15c: r14b, top-k attention k=16                         (D4: auditable attention)
  r15d: [spk, attn, spk] signed + spk_layerscale 0.1       (D2/H1: can spiking layers stack? vs 64.65 / 63.35)
  r15e: tfmr3 + data_independent_windows + Engram v2       (M2: fair transformer baseline)
  r15f: r14b without Engram v2                             (M2: Engram's share of our margin)
  r15g: r15a param-matched by width (n_neurons 5376)       (D1': MLP params -> spiking neurons)"""
import re
D = "/home/glenn/projects/neuro/configs/"
def make(src, name, why, edits, append=""):
    s = open(D + src + ".yaml").read()
    s = re.sub(r"^(# .*\n)+", "", s, count=1)
    s = f"# {name} -- round 15 (2026-10-01). AUTO-GEN temp/mk_gap_r15_configs.py from {src}. Change: {why}\n" + s
    for pat, rep in edits + [(rf"^run_name: {src}$", f"run_name: {name}")]:
        s, n = re.subn(pat, rep, s, flags=re.M)
        assert n == 1, (name, pat)
    s = s.rstrip("\n") + "\n" + append
    open(D + name + ".yaml", "w").write(s)
    print("[OK]", name)
R14B = "screen_r14b_3Lattn2_canon_ramp10_s0_30m"
make(R14B, "screen_r15a_attn_noffn_s0_30m", "attention layers without their MLP (attn_ffn false).", [], "attn_ffn: false\n")
make(R14B, "screen_r15b_spikeglu_s0_30m", "attention-layer GELU MLP -> SpikeGLU (k-WTA 256 of 2752).", [],
     "attn_ffn_spikeglu_k: 256\nattn_ffn_hidden: 2752\n")
make(R14B, "screen_r15c_topk16_s0_30m", "top-k attention (k=16 of 512).", [], "attn_topk: 16\n")
make("screen_r8_signed_3L_s0_30m", "screen_r15d_signed3L_lscale_s0_30m",
     "[spk, attn, spk] + spk_layerscale 0.1 (H1 test; control 64.65 s0 / 63.35 s1).", [], "spk_layerscale: 0.1\n")
make("screen_tfmr3_adamw_s0_30m", "screen_r15e_tfmr3_fair_s0_30m",
     "fair baseline: + data_independent_windows + Engram v2 (as our stacks).",
     [(r"^engram_v2_enabled: false$", "engram_v2_enabled: true")], "data_independent_windows: true\n")
make(R14B, "screen_r15f_r14b_noengram_s0_30m", "Engram v2 off.",
     [(r"^engram_v2_enabled: true$", "engram_v2_enabled: false")])
make(R14B, "screen_r15g_attn_noffn_n5376_s0_30m", "attention layers without MLP, params moved to spiking width (n_neurons 5376).",
     [(r"^n_neurons: 2048$", "n_neurons: 5376")], "attn_ffn: false\n")
