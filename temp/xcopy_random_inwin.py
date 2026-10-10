"""xcopy_random_inwin.py -- 2026-10-06 (Neuro 52, control for E1). Can attention copy a random-token span INSIDE one
chunk? If not, E1's random-span column says nothing about the memory. Also tries random spans drawn from the MQAR
value range (the tokens the memory was trained to store).
  python temp/xcopy_random_inwin.py name=ckpt.pt [...]"""
import sys, argparse, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, MQAR_VAL_LO, MQAR_VAL_HI
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
S, CUE, B, REPS = 32, 4, 16, 4
for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = int(c.get("attn_chunk", 0) or 0) or 512
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"))
    out = []
    for lo, hi, tag in ((1000, c["vocab"] - 1000, "random vocab"), (MQAR_VAL_LO, MQAR_VAL_HI, "MQAR-value range")):
        for where in ("in-window", "beyond", "beyond off"):
            np.random.seed(11); gen = torch.Generator().manual_seed(5); hit = tot = 0
            for _ in range(REPS):
                x, _ = data.batch(B, 2 * C, "val", "cpu")
                src, dst = (100, 300) if where == "in-window" else (C - 64 - S, C + 16)
                x[:, src:src + S] = torch.randint(lo, hi, (B, S), generator=gen)
                x[:, dst:dst + S] = x[:, src:src + S].clone()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = m(x.cuda(), ablate_memory=(where == "beyond off")).argmax(-1).cpu()
                hit += (pred[:, dst + CUE - 1:dst + S - 1] == x[:, dst + CUE:dst + S]).sum().item()
                tot += B * (S - CUE)
            out.append(f"{tag} {where} {100.0 * hit / tot:5.1f}")
    print(f"{name:26s} C={C} E1-ctrl random-span copy | " + " | ".join(out), flush=True)
    del m; torch.cuda.empty_cache()
