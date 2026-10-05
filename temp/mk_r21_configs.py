"""mk_r21_configs.py -- 2026-10-04 (Glenn: cancel h2, investigate the G3 hippocampus collapse thoroughly; v4 demo in
2 weeks). G3 shape (d 768, 12 layers, v4 corpus/vocab), 200M tokens per run.
Finding that motivates it: in g3_cand_150m the H0 memory is never used (memory delta <= 0 from eval 1, write gates 97%
at floor, spikes fade 0.28 -> 0.004) while recall stays 100% -> with 11 full-window attention layers, attention does all
in-window work. At 30M the late gap dose (x1, r16) gave haystack 100% at every gap for +2.4% ppl.
  r21a  control (g3_cand @200M)
  r21b  + x1 late gap dose (gap MQAR up to 440 from 70% of training, w 0.15, batch 16)
  r21c  + fixed 128-token attention window (train + eval) + x1 -> recall beyond 128 tokens must go through memory
  r21f  + fixed 128 window, no gap dose (window effect alone)
  r21d  fair transformer (g3_tfmr) + x1
  r21e  fair transformer + fixed 128 window + x1 (windowed, memory-free baseline)
Gap MQAR sequences are up to 2*32+1+440 = 505 tokens, so gap arms pad MQAR to 512 (costs speed; screens only)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
B = dict(max_tokens=200_000_000, eval_every=1000, ckpt_every=1000)
X1 = dict(mqar_gap_max=440, mqar_gap_start_frac=0.7, mqar_gap_ramp_frac=0.1, mqar_gap_weight=0.15, mqar_batch=8,
          mqar_pad_to=65, mqar_gap_pad_to=512, mqar_compile_static=True, gdn2_cumsum_eager_max_b=8)
W = dict(attn_window_fixed=128)


def emit(src, name, note, **upd):
    c = yaml.safe_load(open(R + src + ".yaml")); c.update(B, **upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r21_configs.py (2026-10-04) from {src}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)


emit("g3_cand_150m", "r21a_ctrl_150m", "Control at 200M tokens.")
emit("g3_cand_150m", "r21b_x1_150m", "+ late gap dose (x1).", **X1)
emit("g3_cand_150m", "r21c_w128x1_150m", "+ fixed 128 attention window + x1.", **W, **X1)
emit("g3_cand_150m", "r21f_w128_150m", "+ fixed 128 attention window, no gap dose.", **W)
emit("g3_tfmr_150m", "r21d_tfmr_x1_150m", "Fair transformer + x1.", **X1)
emit("g3_tfmr_150m", "r21e_tfmr_w128x1_150m", "Fair transformer + fixed 128 window + x1.", **W, **X1)
