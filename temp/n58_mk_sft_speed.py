"""n58_mk_sft_speed.py -- 2026-10-12 (Neuro 58; demo plan step 1). configs/sft_v4_dense_speed.yaml: the v4 SFT stage on
the DENSE 1B architecture (base = the newest pulled v4_cloud checkpoint in models/checkpoints/v4_cloud_pin), sized
for the 5090 (B4 x GA8 x 512 = 16,384 tok/step as in temp/mk_sft_v4.py), with the 13L-rehearsed v2 blend. Only 1.5M
tokens: it MEASURES tok/s and peak memory for the SFT-A / SFT-B schedule; the result is not a model to keep."""
import yaml
R = "/home/glenn/projects/neuro/configs/"; PT = "/home/glenn/projects/neuro/training_data/posttrain/"
c = yaml.safe_load(open(R + "v4_cloud.yaml"))
ref = yaml.safe_load(open(R + "sft_v4_reh13_v2tr.yaml"))
for k in ("head_weight_decay", "kd_enabled", "kd_teacher", "kd_alpha", "kd_temperature", "kd_compile", "kd_compile_mode", "no_decay_names"):
    c.pop(k, None)
for k in ("sft_jsonl", "sft_dataset", "sft_split", "sft_dataset_frac", "sft_max_len", "lut_subword_fallback", "lora_enabled",
          "gen_check_prompt", "mqar_aug", "mqar_N_min", "mqar_N_max", "mqar_weight", "weight_decay", "lr", "muon_lr",
          "lr_min_frac", "lr_warmup_frac", "seed", "tool_proto", "sft_tools_full_frac"):
    if k in ref: c[k] = ref[k]
c.update(run_name="sft_v4_dense_speed", sft_init="v4_cloud_pin", batch_size=4, grad_accum=8, max_tokens=1_500_000,
         ckpt_every=100000, gen_check_every=100000, log_every=5, keep_last=1, eval_every=100000,
         mqar_batch=1, mqar_xchunk_batch=1, mqar_aug_batch=int(ref.get("mqar_aug_batch", 8)),
         data_bin="/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin")
assert c["batch_size"] * c["grad_accum"] * c["sft_max_len"] == 16384
with open(R + "sft_v4_dense_speed.yaml", "w") as f:
    f.write("# sft_v4_dense_speed -- AUTO-GEN temp/n58_mk_sft_speed.py (2026-10-12): 1B dense SFT speed / memory measurement on the 5090.\n")
    yaml.safe_dump(c, f, sort_keys=False)
print("[OK] sft_v4_dense_speed.yaml | mqar_aug_batch", c["mqar_aug_batch"], "| jsonl", [p.split("/")[-1] for p in c["sft_jsonl"]])
