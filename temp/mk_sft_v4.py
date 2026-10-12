"""mk_sft_v4.py -- 2026-10-11 (Neuro 56). SFT configs for the v4 architecture (RC3 + spike codes).
  configs/sft_v4_reh13.yaml   home rehearsal: 13L, base = the r26o conversion checkpoint, 12M tok
  configs/sft_v4_cloud.yaml   the real stage: 1B, base = v4_cloud_conv (latest checkpoint), 60M tok
ARCH = every key of the base's OWN config (its conversion config), so src/train_sft.py -- which builds the model with
train_gpu.build_model -- builds exactly the model the checkpoint holds, spike codes included.
SFT KEYS = configs/sft_v3_363m.yaml, the stage that produced the v3 demo candidate (tool .96): blended tools +
coldstart + 50% alpaca, max_len 512, lr 1e-4, muon_lr 1e-3, MQAR rehearsal at weight 0.1, 16,384 tok/step, 60M tok.
Not carried over from the base: head_weight_decay (guards a long cosine run; no validated SFT config has it),
weight_decay (0 in SFT), the KD keys (train_sft has no teacher). Keys train_sft ignores (data_bin, schedule fractions,
compile mode, sparsity loss) stay as in the base: the eval scripts read data_bin / data_prelut from the checkpoint.
  python temp/mk_sft_v4.py        [SFT_CLOUD_B=16 SFT_CLOUD_GA=2: 1B on the H200;  4 / 8 on the 5090]"""
import os, yaml
R = "/home/glenn/projects/neuro/configs/"
PT = "/home/glenn/projects/neuro/training_data/posttrain/"
SFT = dict(
    sft_jsonl=[PT + "tools.jsonl", PT + "coldstart.jsonl"], sft_dataset="yahma/alpaca-cleaned", sft_split="train",
    sft_dataset_frac=0.5, sft_max_len=512, lut_subword_fallback=True, lora_enabled=False,
    gen_check_prompt="### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    mqar_aug=True, mqar_N_min=4, mqar_N_max=32, mqar_weight=0.1,
    weight_decay=0.0, lr=1.0e-4, muon_lr=1.0e-3, lr_min_frac=0.1, lr_warmup_frac=0.05, seed=0)
DROP = ("head_weight_decay", "kd_enabled", "kd_teacher", "kd_alpha", "kd_temperature", "kd_compile", "kd_compile_mode")


def make(base_cfg, name, init, note, **upd):
    c = yaml.safe_load(open(R + base_cfg + ".yaml"))
    for k in DROP:
        c.pop(k, None)
    c.update(SFT)
    c.update(run_name=name, sft_init=init, **upd)
    assert c["batch_size"] * c["grad_accum"] * c["sft_max_len"] == 16384, "keep 16,384 tok/step (the v3 SFT value)"
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_sft_v4.py (2026-10-11). {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    steps = c["max_tokens"] // 16384
    print(f"[OK] {name}.yaml: base {init}, B{c['batch_size']} x GA{c['grad_accum']}, {c['max_tokens'] / 1e6:.0f}M tok = {steps} steps")


make("r26o_13L_RC3_conv_llamakd_150m", "sft_v4_reh13", "r26o_13L_RC3_conv_llamakd_150m",
     "Home REHEARSAL of the v4 SFT stage on the 13L RC3 conversion checkpoint (12M tok).",
     batch_size=4, grad_accum=8, max_tokens=12_000_000, ckpt_every=250, gen_check_every=125, log_every=25, keep_last=2)
make("v4_cloud_conv", "sft_v4_cloud", "v4_cloud_conv",
     "v4 SFT stage at 1B: run ONLY after v4_cloud_conv is DONE (sft_init takes its LATEST checkpoint).",
     batch_size=int(os.environ.get("SFT_CLOUD_B", 16)), grad_accum=int(os.environ.get("SFT_CLOUD_GA", 2)),
     max_tokens=60_000_000, ckpt_every=250, gen_check_every=250, log_every=50, keep_last=3)
