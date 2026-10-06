"""Derive the post-training SCREEN configs from the REAL configs so they never
drift. Overrides:
  - run_name: distinct, so the screen never touches the future final run's ckpts.
  - init pointer: frozen partial base for SFT; the SFT-screen output for ORPO.
  - 5080 VRAM fit (2026-09-21): the real configs (batch 4, compile:true,
    gdn2_ckpt_chunks:false) OOM'd on the 16GB 5080 -- the fixed cost (model +
    optimizer + compile/autotune workspace) alone is ~12.7GB, leaving no room
    for batch-4 activations. Fix without touching the real configs: compile off
    (drops the big fixed workspace), gradient-checkpoint the GDN-2 chunks
    (activation saver), half the micro-batch with matching grad_accum so the
    EFFECTIVE batch is unchanged, and a smaller eval batch. Speed cost is fine:
    this is a method screen, not a throughput run.
"""
import yaml

FIT_5080 = {              # shared VRAM-fit overrides for the 16GB 5080
    "compile": False,
    "gdn2_ckpt_chunks": True,
    "eval_batch": 4,
}

def patch(src, dst, overrides):
    with open(src) as f:
        c = yaml.safe_load(f)
    c.update(overrides)
    with open(dst, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False)
    print(f"[mk_screen] wrote {dst} (run_name={c['run_name']}, "
          f"batch={c.get('batch_size')}x ga{c.get('grad_accum')}, "
          f"compile={c.get('compile')}, ckpt_chunks={c.get('gdn2_ckpt_chunks')})")

# SFT: real is batch 4 x ga 8 (eff 32) -> screen batch 1 x ga 32 (eff 32).
# batch 2 peaked at 15.9/16.3GB on the 5080 (eval spikes would OOM); batch 1
# roughly halves activations for comfortable headroom on the unattended run.
patch("configs/sft_v3_363m.yaml", "temp/sft_v3_363m_screen.yaml",
      {"run_name": "sft_v3_363m_screen",
       "sft_init": "flagship_v3_363m_frozen_screen",
       "batch_size": 1, "grad_accum": 32, **FIT_5080})

# ORPO: real is batch 4 x ga 4 (eff 16 pairs) -> screen batch 1 x ga 16 (eff 16).
patch("configs/orpo_v3_363m.yaml", "temp/orpo_v3_363m_screen.yaml",
      {"run_name": "orpo_v3_363m_screen",
       "dpo_init": "sft_v3_363m_screen",
       "batch_size": 1, "grad_accum": 16, **FIT_5080})
