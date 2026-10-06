"""diag_key_similarity.py -- 2026-09-29: do unsigned spike codes make the GDN memory keys overlap (interfere)?
Hypothesis: spikes >= 0 give every key a shared mean component (W_k @ mean activity) -> high pairwise cosine ->
writes of ordinary text overwrite stored facts -> the decay gates learn to wipe. Signed spikes can cancel it.
Hooks the first spiking block's W_k / W_q on real val text; reports the mean |cosine| between keys at random
position pairs, the share of key energy in the mean direction, and the same after mean-centering.
  CUDA_VISIBLE_DEVICES=1 python temp/diag_key_similarity.py name=ckpt [...]"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
torch.set_float32_matmul_precision("high")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    blk = [b for b in m.blocks if hasattr(b, "W_k")][0]
    cap = {}
    blk.W_k.register_forward_hook(lambda mod, i, o: cap.__setitem__("k", o.detach().float()))
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    np.random.seed(0); x, _ = data.batch(16, 512, "val", "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    k = cap["k"].reshape(-1, cap["k"].shape[-1])[1:]          # (B*T, dk)
    k = k[k.norm(dim=-1) > 0]
    g = torch.Generator(device="cuda").manual_seed(0)
    i = torch.randint(0, k.shape[0], (20000,), device="cuda", generator=g); j = torch.randint(0, k.shape[0], (20000,), device="cuda", generator=g)
    cos = F.cosine_similarity(k[i], k[j], dim=-1)
    mu = k.mean(0)
    share = ((k @ mu / mu.norm()) ** 2).sum() / (k ** 2).sum()
    kc = k - mu
    cosc = F.cosine_similarity(kc[i], kc[j], dim=-1)
    print(f"{name:12s} key |cos| mean {cos.abs().mean():.3f} (signed mean {cos.mean():+.3f}) | energy in mean direction "
          f"{share:.1%} | after centering |cos| {cosc.abs().mean():.3f}", flush=True)
    del m; torch.cuda.empty_cache()
print("random 1024-d vectors: |cos| ~0.025")
