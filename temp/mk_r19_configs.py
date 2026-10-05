"""mk_r19_configs.py -- 2026-10-03 (Glenn: "best of both worlds" -- GELU-level ppl/haystack plus sparsity,
interpretability, energy; membranes stay in the hippocampal layer). Round 19 = l3 (GELU MLP, 2 seeds: 40.13/40.41)
with the MLP activation swapped for relu2 (Primer; "ReLU^2 Wins" 2024), optionally with a homeostatic firing
threshold (SparseAct, src/hybrid_attn.py): dense until 60% of training, then the rate target ramps geometrically to
the target by 90% (same late schedule as the DSN k-ramp). Compare with r17k DSN 41.17 @ 25% active (3 seeds).
Only the MLP flags differ from screen_r16_l3_gelu_s0_30m."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
SRC = "screen_r16_l3_gelu_s0_30m"
H = dict(attn_ffn_act="relu2", attn_ffn_ramp_start=0.6, attn_ffn_ramp_end=0.9, attn_ffn_homeo_lr=0.02)
ARMS = {
    "r19a_relu2": ("relu2, natural sparsity (no threshold)", dict(attn_ffn_act="relu2")),
    "r19b_layer25": ("relu2 + per-layer homeostatic threshold -> 25% active (DSN r17k-matched)",
                     dict(H, attn_ffn_homeo="layer", attn_ffn_rate_target=0.25)),
    "r19c_layer10": ("relu2 + per-layer threshold -> 10% active", dict(H, attn_ffn_homeo="layer", attn_ffn_rate_target=0.10)),
    "r19d_unit10": ("relu2 + per-neuron threshold -> 10% active (every neuron fires ~10%)",
                    dict(H, attn_ffn_homeo="unit", attn_ffn_rate_target=0.10)),
    "r19e_layer05": ("relu2 + per-layer threshold -> 5% active", dict(H, attn_ffn_homeo="layer", attn_ffn_rate_target=0.05)),
}
for arm, (note, upd) in ARMS.items():
    c = yaml.safe_load(open(R + SRC + ".yaml"))
    name = f"screen_{arm}_s0_30m"
    c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r19_configs.py (2026-10-03) from {SRC}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)
