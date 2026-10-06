"""mk_gap_r17_configs.py -- 2026-10-02 round 17 (Glenn OK 10-02: "pursue the DSN trade-off with config tweaks").
Config-only screens to recover the DSN's +9% cost (c0 2-seed 43.61 vs l3 GELU 40.13; the gap opens only during the
late k-ramp, docs/to-do.html r16 entry). All from screen_r16_c0_s0_30m (4L [H, (A+DSN)x3]); run on the 5090.
  f: k 688/2752 (25% active instead of 9.3%)
  g: k-ramp starts at 30% instead of 60% (more training time at the sparse budget)
  h: wider DSN hidden 4096 with k 384 (same ~9.4% active; +12.5M params, more neurons per active one)
  i: k 512 (18.6%) + ramp start 0.4 (middle ground)
  j: X2 DSN-LIF with sparsity target 0.20 (threshold neurons allowed to fire ~2x more; also applies to H0)"""
import re
D = "/home/glenn/projects/neuro/configs/"
def make(src, name, why, edits):
    s = open(D + src + ".yaml").read()
    s = re.sub(r"^(# .*\n)+", "", s, count=1)
    s = f"# {name} -- round 17 (2026-10-02). AUTO-GEN temp/mk_gap_r17_configs.py from {src}. Change: {why}\n" + s
    for pat, rep in edits + [(rf"^run_name: {src}$", f"run_name: {name}")]:
        s, n = re.subn(pat, rep, s, flags=re.M)
        assert n == 1, (name, pat, n)
    open(D + name + ".yaml", "w").write(s)
    print("[OK]", name)
C0, X2 = "screen_r16_c0_s0_30m", "screen_r16_x2_dsnlif_s0_30m"
make(C0, "screen_r17f_k688_s0_30m", "DSN k 688 (25% active).", [(r"^attn_ffn_spikeglu_k: 256$", "attn_ffn_spikeglu_k: 688")])
make(C0, "screen_r17g_ramp03_s0_30m", "DSN k-ramp from 30% of training.",
     [(r"^attn_ffn_spikeglu_ramp_start: 0.6$", "attn_ffn_spikeglu_ramp_start: 0.3")])
make(C0, "screen_r17h_wide4096_s0_30m", "DSN hidden 4096, k 384 (same ~9.4% active, wider).",
     [(r"^attn_ffn_spikeglu_k: 256$", "attn_ffn_spikeglu_k: 384"), (r"^attn_ffn_hidden: 2752$", "attn_ffn_hidden: 4096")])
make(C0, "screen_r17i_k512_ramp04_s0_30m", "DSN k 512 (18.6%) + ramp from 40%.",
     [(r"^attn_ffn_spikeglu_k: 256$", "attn_ffn_spikeglu_k: 512"),
      (r"^attn_ffn_spikeglu_ramp_start: 0.6$", "attn_ffn_spikeglu_ramp_start: 0.4")])
make(X2, "screen_r17j_x2_target20_s0_30m", "DSN-LIF with sparsity target 0.20 (also H0).",
     [(r"^sparsity_target: 0.10$", "sparsity_target: 0.20")])
