"""test_muon_attn_lr.py: default (no muon_attn_lr) grouping/steps identical to HEAD; with it, attn matrices get their own lr."""
import sys, os, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
if os.environ.get("GAP_STAGE"): sys.path.insert(0, os.environ["GAP_STAGE"])
from train_gpu import build_model, _build_optimizers
import muon; print("muon module:", muon.__file__)
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_honest_hybrid_s0_30m.yaml"))
def steps(cfg):
    torch.manual_seed(0); m = build_model(cfg).to("cuda"); opts = _build_optimizers(m, cfg, "cuda")
    x = torch.randint(0, cfg["vocab"], (2, 257), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x[:, :-1]).float().reshape(-1, cfg["vocab"]), x[:, 1:].reshape(-1))
        loss.backward(); [o.step() for o in opts]; [o.zero_grad() for o in opts]
    return m, opts
m1, o1 = steps(c)
print("default groups:", [(len(g["params"]), g["lr"]) for g in o1[0].param_groups])
torch.save({k: v.cpu() for k, v in m1.state_dict().items()}, "/tmp/muon_attn_ref.pt") if sys.argv[1] == "save" else None
if sys.argv[1] == "check":
    ref = torch.load("/tmp/muon_attn_ref.pt")
    d = max((ref[k].float() - v.cpu().float()).abs().max().item() for k, v in m1.state_dict().items())
    print("default path vs HEAD after 3 steps: max |diff| =", d)
    m2, o2 = steps(dict(c, muon_attn_lr=0.02))
    print("attn-lr groups:", [(len(g["params"]), g["lr"]) for g in o2[0].param_groups])
