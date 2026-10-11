"""mk_convsmoke.py -- 2026-10-11 (Neuro 55). The 1B conversion smoke (configs/v4_smoke1b_conv.yaml, T2048) dies at its
first forward on the 5090 (WDDM cliff: 1B model + Llama teacher + spike codes). Two smaller variants, both resumed from
the SAME seeded checkpoint (step 150, 614,400 tok) that temp/mk_spike_seed.py wrote from the 1B dense smoke:
  v4_smoke1b_conv_t1024 = the conversion config at T1024 / attn_chunk 512 (half the activations, every code path kept).
      Step and token counters come from the ckpt, so max_tokens = 2048 x 900 -> runs steps 150 -> 750 of 900 (17% -> 83%):
      plain MQAR, sparsity ramp from 60%, gap + xchunk MQAR from 70% (step 630).
  v4_smoke1b_conv_noaux = the real T2048 shape with the MQAR aux off: does the main conversion path fit at all?"""
import yaml
R = "/home/glenn/projects/neuro/configs/"
k = yaml.safe_load(open(R + "v4_smoke1b_conv.yaml"))
for name, note, upd in (("v4_smoke1b_conv_t1024", "T1024 / attn_chunk 512, steps 150 -> 750 of 900.",
                         dict(block_size=1024, attn_chunk=512, max_tokens=2048 * 900)),
                        ("v4_smoke1b_conv_noaux", "real T2048 shape, MQAR aux off, steps 150 -> 210.",
                         dict(mqar_aug=False))):
    c = dict(k, **upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_convsmoke.py (2026-10-11). 5090 conversion smoke at 1B: {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("wrote", name)
