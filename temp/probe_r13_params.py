# probe_r13_params.py -- 2026-09-30 CPU-only: param counts to size the r13 matched-compute width control (D)
import sys, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_r5_signed_s0_30m.yaml"))
three = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_r8_signed_3L_s0_30m.yaml"))
def count(c, emb_too=False):
    m = build_model(c)
    tot = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters() if "embed" in n or "engram" in n or n.startswith("head"))
    return tot / 1e6, (tot - emb) / 1e6
print("2L signed", count(base))
print("3L signed", count(three))
a = dict(three); a["attention_layer_positions"] = [1, 2]
print("3L [spk,attn,attn]", count(a))
for n in (2560, 2816, 3072, 3328):
    c = dict(base); c["n_neurons"] = n
    print("2L n_neurons", n, count(c))
