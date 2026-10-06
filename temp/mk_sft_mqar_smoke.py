# mk_sft_mqar_smoke.py -- 2026-09-26: smoke config for train_sft's new mqar_aug branch.
# 30M spiking ckpt (screen_bs512_s0_30m) as sft_init, coldstart.jsonl only, 6 steps.
import yaml, pathlib
c = yaml.safe_load(open("configs/screen_bs512_s0_30m.yaml"))
c.update(run_name="smoke_sft_mqar_30m", sft_init="screen_bs512_s0_30m",
         sft_jsonl=["/home/glenn/projects/neuro/training_data/posttrain/coldstart.jsonl"],
         sft_max_len=512, lut_subword_fallback=True, gen_check_every=0, lora_enabled=False,
         mqar_aug=True, mqar_weight=0.5, batch_size=4, grad_accum=2,
         max_tokens=4 * 2 * 512 * 6, log_every=1, ckpt_every=10**9, keep_last=1,
         lr=0.0001, muon_lr=0.001, weight_decay=0.0, lr_warmup_frac=0.05, lr_min_frac=0.1)
pathlib.Path("temp/smoke_sft_mqar_30m.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
print("wrote temp/smoke_sft_mqar_30m.yaml")
