"""mk_v4_cloud.py -- 2026-10-09 (Neuro 54). The v4 cloud configs = RC3 (r26n dense -> r26o conversion) at the 1B shape.
  configs/v4_cloud.yaml       dense phase: every RC3 key of r26n, with the shape / data keys of v4_cloud_bench (1B).
  configs/v4_cloud_conv.yaml  conversion phase: the r26o recipe (L8 signed + q/v spike codes, Llama-3.2-1B KD), seeded
                              from the last dense ckpt by temp/mk_spike_seed.py into its OWN run dir (dense ckpt kept).
DRAFT until the H200 re-bench and Glenn's token budget: DENSE_TOK / CONV_TOK / batch below are placeholders.
  python temp/mk_v4_cloud.py [DENSE_TOK] [CONV_TOK]"""
import sys, yaml
R = "/home/glenn/projects/neuro/configs/"
DENSE_TOK = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000_000
CONV_TOK = int(float(sys.argv[2])) if len(sys.argv) > 2 else 4_000_000_000   # r26o ratio: 80M on 200M = 40%
rc3 = yaml.safe_load(open(R + "r26n_13L_RC2_xlate_150m.yaml"))
bench = yaml.safe_load(open(R + "v4_cloud_bench.yaml"))
conv = yaml.safe_load(open(R + "r26o_13L_RC3_conv_llamakd_150m.yaml"))

c = dict(rc3)
# 1B shape (probe_g4_r19d via v4_cloud_bench); attn_ffn_homeo stays "layer" (Glenn 10-04, r19e) as in every r26 run.
for k in ("d", "n_neurons", "d_mem", "attention_heads", "attn_ffn_hidden"):
    c[k] = bench[k]
# 32,768 tok/step as in the H200 bench (B64 x T512 there). RC3 aux doses per LM token kept: gap / xchunk batch = B,
# plain MQAR batch = 4 x B (r26n: 4 / 4 / 16 at B4).
# 2026-10-11: batch split from the command line (argv 3 / 4). B x GA must stay 16 = 32,768 tok/step; the aux doses
# below follow B per micro-step, so the dose per LM token is the same for any split (r26n: B4 x GA2).
# DEFAULT B8 x GA2 (was B16 x GA1): 1B memory probe on the 5090 (logs/mem1b.log, B1): 13.4 GB held + 8.3 GB per
# T2048 sample, + 3.2 GB per sample on a cross-chunk step (conversion: 15.7 + ~8.9 + ~3.5). B16 = 163-197 GB on a
# cross-chunk step vs 143 GB on the H200 -> it would have died at the FIRST xchunk batch, 70% into the dense phase.
# B8 = 89-105 GB (dense), 97-115 GB (conversion). cloud/bench_rc3.sh measures the real peak and picks the split.
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GA = int(sys.argv[4]) if len(sys.argv) > 4 else 16 // B
assert B * GA == 16, 'B x GA must be 16 (32,768 tok/step)'
c.update(batch_size=B, grad_accum=GA, eval_batch=2, mqar_batch=B, mqar_xchunk_batch=B, mqar_aug_batch=4 * B)
c.update(run_name="v4_cloud", max_tokens=DENSE_TOK,
         data_bin="/workspace/data/tokenized_v4/train_lut32768_u16.bin", data_prelut=True, lut_subword_fallback=True,
         eval_every=3000, ckpt_every=1500, keep_last=3)        # ~1 step/s on the H200 -> a ckpt every ~25 min
# Long-run guard (Glenn 10-09): the val_ppl tripwire alone, with the flagship_v3 tuning (the gate tripwires false-trip
# on RC3). r26n's val series does not trip it. Its state is not checkpointed: it restarts after every resume.
# head_weight_decay 1.0 = ON (Glenn 10-10): insurance against the late LM-head runaway of flagship v1 (v3 ran with it)
# on a ~300k-step run with no slack. r26p (150M): +2.2% chunk1, inside the ~2% same-seed noise. Not resume-compatible:
# it must be in from step 0.
c.update(head_weight_decay=1.0)
c.update(watchdog_enabled=False, watchdog_valppl_only=True, watchdog_valppl_ema_alpha=0.4, watchdog_valppl_tol=0.07,
         watchdog_valppl_patience=2)
with open(R + "v4_cloud.yaml", "w") as f:
    f.write("# v4_cloud -- AUTO-GEN temp/mk_v4_cloud.py (2026-10-09). RC3 dense phase (r26n) at the 1B shape. DRAFT: "
            "token budget + batch pending the H200 re-bench.\n")
    yaml.safe_dump(c, f, sort_keys=False)

# Conversion: every key r26o changed vs r26n, except the home batch split (B2xGA4 was for the 5090's VRAM).
k = dict(c)
k.update({key: v for key, v in conv.items() if rc3.get(key) != v and key not in
          ("run_name", "max_tokens", "batch_size", "grad_accum", "eval_batch", "mqar_batch", "mqar_xchunk_batch",
           "mqar_aug_batch")})
# The Llama teacher needs RAW Llama-3 ids; the uint16 corpus has none (3.66% of its tokens are id 0 = OOV, so an
# inverse LUT would corrupt the teacher's context). -> a uint32 slice of the raw corpus + the val tail (temp/mk_kd_slice.py,
# not written yet), read without prelut.
# 2026-10-11: the conversion phase may need a smaller batch than the dense phase (spike codes + the Llama teacher cost
# ~1.5x the activation memory per token: 1B smoke on the 5090) -> its own split from argv 5 (default = the dense B).
CB = int(sys.argv[5]) if len(sys.argv) > 5 else B
assert 16 % CB == 0
k.update(batch_size=CB, grad_accum=16 // CB, mqar_batch=CB, mqar_xchunk_batch=CB)
k.update(run_name="v4_cloud_conv", max_tokens=DENSE_TOK + CONV_TOK, mqar_aug_batch=2 * CB,   # r26o: 4 at B2
         data_bin="/workspace/data/tokenized_v4/train_kd_slice_u32.bin", data_prelut=False)
with open(R + "v4_cloud_conv.yaml", "w") as f:
    f.write("# v4_cloud_conv -- AUTO-GEN temp/mk_v4_cloud.py (2026-10-09). RC3 conversion phase (r26o: spike codes + "
            "Llama-3.2-1B KD) from the last v4_cloud ckpt. DRAFT.\n")
    yaml.safe_dump(k, f, sort_keys=False)
# Home dress rehearsal on the 5080 (temp/queue_5080_reh.sh runs cloud/run_v4.sh on these): the SAME two configs scaled
# to the 7L / d768 screen shape, 40M + 16M tok (the 40% ratio), home copies of the SAME two data files.
T4 = "/home/glenn/projects/bdh/data/tokenized_v4/"
REH = dict(d=768, n_neurons=3072, d_mem=768, attention_heads=12, n_layers=7, attention_layer_positions=[0, 1, 2, 4, 5, 6],
           batch_size=2, grad_accum=4, eval_batch=1, mqar_batch=2, mqar_xchunk_batch=2, eval_every=500, ckpt_every=500)
for src, name, upd in ((c, "v4_reh", dict(max_tokens=40_000_000, mqar_aug_batch=8,
                                           data_bin=T4 + "train_lut32768_u16.bin")),
                       # 10-10 00:45: at B2xGA4 the Llama teacher spilled the 16 GB 5080 (reserved 16.2 GB, log silent
                       # 28 min after step 2500 -> watchdog kill) -> B1xGA8 for the home rehearsal only.
                       (k, "v4_reh_conv", dict(max_tokens=56_000_000, mqar_aug_batch=2, batch_size=1, grad_accum=8,
                                                mqar_batch=1, mqar_xchunk_batch=1,
                                                data_bin=T4 + "train_kd_slice_u32.bin"))):
    r = dict(src, **{**REH, **upd}, run_name=name); r.pop("attn_ffn_hidden")
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_v4_cloud.py (2026-10-09). 5080 dress rehearsal of {src['run_name']} "
                "(7L, 40M + 16M tok).\n")
        yaml.safe_dump(r, f, sort_keys=False)
# 1B SMOKE on the 5090 (Glenn 10-10; temp/queue_5090_smoke1b.sh runs cloud/run_v4.sh on these): the REAL cloud configs
# at the 1B shape, only batch (B1xGA2 = 4,096 tok/step), budget (150 + 60 steps), eval / ckpt cadence and home data
# paths differ. Covers at 1B: build + compile, xchunk from 50%, gap MQAR from 70%, sparsity ramp, eval, ckpt, a planned
# stop + resume, mk_spike_seed, Llama KD conversion. Not a quality or a speed number.
SMK = dict(batch_size=1, grad_accum=2, eval_batch=1, mqar_batch=1, mqar_xchunk_batch=1, eval_every=50, ckpt_every=50,
           log_every=10, keep_last=2, eval_iters=10)
for src, name, upd in ((c, "v4_smoke1b", dict(max_tokens=4096 * 150, mqar_aug_batch=4,
                                               data_bin=T4 + "train_lut32768_u16.bin")),
                       (k, "v4_smoke1b_conv", dict(max_tokens=4096 * 210, mqar_aug_batch=2,
                                                    data_bin=T4 + "train_kd_slice_u32.bin"))):
    r = dict(src, **{**SMK, **upd}, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_v4_cloud.py (2026-10-10). 5090 smoke of {src['run_name']} at the 1B shape "
                "(150 + 60 steps).\n")
        yaml.safe_dump(r, f, sort_keys=False)
print(f"[OK] v4_cloud.yaml ({DENSE_TOK / 1e9:.1f}B tok, {DENSE_TOK // (B * GA * c['block_size'])} steps, B{B} x GA{GA}) + v4_cloud_conv.yaml "
      f"(+{CONV_TOK / 1e9:.1f}B)")
print("conv keys vs dense:", {key: k[key] for key in k if c.get(key) != k[key]})
