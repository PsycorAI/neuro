#!/usr/bin/env python3
# mk_probe_neuron_ab.py -- 2026-09-30: 1B B8 neuron A/B probe configs (docs/fused_neuron_kernel_design.md §5A / §7,
# docs/cuda_graphs_v4_design.md §5 step 0). From temp/probe_1b_b8.yaml (FPT, 3,880-4,203 tok/s on 09-26), ONE change per arm:
#   fpt   = as-is (lif_parallel K=10)
#   lifml = lif_exact: true, lif_levels: 4, lif_signed: true (fused exact neuron, current best 30M stack)
# 100 optimizer steps each (36,864 tok/step); steady-state window = steps >= 50 (after compile + MQAR recompiles).
import pathlib, yaml
base = yaml.safe_load(open("/home/glenn/projects/neuro/temp/probe_1b_b8.yaml"))
assert not base.get("lif_exact") and base["batch_size"] == 8 and base["grad_accum"] == 9
for arm, over in [("fpt", {}), ("lifml", {"lif_exact": True, "lif_levels": 4, "lif_signed": True})]:
    for rep in (1, 2):
        c = dict(base); c.update(over)
        c.update(run_name=f"probe_1b_b8_{arm}_r{rep}", max_tokens=36864 * 100, log_every=10)
        p = pathlib.Path(f"/home/glenn/projects/neuro/temp/probe_1b_b8_{arm}_r{rep}.yaml")
        p.write_text(f"# AUTO-GEN temp/mk_probe_neuron_ab.py -- 1B B8 neuron A/B, arm {arm}, repeat {rep}\n"
                     + yaml.safe_dump(c, sort_keys=False))
        print("wrote", p)
