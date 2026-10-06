"""check_causal_ckpt.py -- 2026-09-29: causality check on a trained ckpt. Change the input token at position p;
logits at every position < p must be bit-identical (any change = future-token leak). Eval mode, bf16 autocast."""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval()
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    torch.manual_seed(0)
    x, _ = data.batch(4, 512, "val", "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = m(x).float()
        for p in (1, 100, 255, 256, 300, 511):
            x2 = x.clone(); x2[:, p] = (x2[:, p] + 1234) % c["vocab"]
            out = m(x2).float()
            before = (out[:, :p] - ref[:, :p]).abs().max().item()
            at = (out[:, p:] - ref[:, p:]).abs().max().item()
            print(f"{name}: change token@{p:3d} -> max|dlogit| before p = {before:.3e} (must be 0) | at/after p = {at:.2f}")
