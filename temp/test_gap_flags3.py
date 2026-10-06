"""test_gap_flags3.py: canon_a identity at init + grads; mqar_gap_batch shapes / no key clash / target correct."""
import sys, os, yaml, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
if os.environ.get("GAP_STAGE"): sys.path.insert(0, os.environ["GAP_STAGE"])
import train_gpu as T
from train_gpu import build_model, StreamText, BIN
base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_gap_hyb_full_s0_30m.yaml"))
torch.manual_seed(0); ctl = build_model(base).to("cuda").eval()
x = torch.randint(0, 16384, (2, 257), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
m = build_model(dict(base, canon_a=4)).to("cuda"); miss, _ = m.load_state_dict(ctl.state_dict(), strict=False); m.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    d = (m(x[:, :-1]).float() - ctl(x[:, :-1]).float()).abs().max().item()
m.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = F.cross_entropy(m(x[:, :-1]).float().reshape(-1, 16384), x[:, 1:].reshape(-1))
loss.backward()
print(f"canon_a: new params {miss} | init logits diff {d:.2e} | grad norm {m.blocks[0].canon_a_w.grad.norm().item():.3e}")
data = StreamText(BIN, vocab=16384, orig_vocab=128256); rng = np.random.RandomState(0)
for N, g in ((4, 0), (8, 100), (32, 440)):
    xb, tg = T.mqar_gap_batch(16, N, g, data, rng, "cpu")
    keys = xb[:, 0:2 * N:2]; vals = xb[:, 1:2 * N:2]; q = xb[:, -1]
    pos = (keys == q.unsqueeze(1)).float().argmax(1)
    ok_t = torch.equal(vals[torch.arange(16), pos], tg)
    gap = xb[:, 2 * N:-1]
    clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any().item()
    print(f"N{N} g{g}: shape {tuple(xb.shape)} (expect {(16, 2 * N + g + 1)}) | target ok {ok_t} | gap-key clash {clash} | gap sample {gap[0, :8].tolist()}")
