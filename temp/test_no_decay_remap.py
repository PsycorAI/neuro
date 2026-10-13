"""test_no_decay_remap.py -- 2026-10-12 (Neuro 58). CPU test of no_decay_names + muon.remap_adamw_state.
1. default (no_decay_names None): grouping identical to before (embedding in the decay group).
2. with ["embed.weight"]: embedding in the no-decay group, same number of groups, every other tensor where it was.
3. resume across the switch: train 5 steps old layout, save state, build the new layout, remap, load ->
   each parameter's moments equal the saved ones; 300 more steps: the embedding norm no longer shrinks under zero
   gradient while a decayed matrix still does."""
import sys, copy, torch, torch.nn as nn
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from muon import build_muon_adamw, remap_adamw_state


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 16); self.bucket = nn.Embedding(40, 16)
        self.fc = nn.Linear(16, 16); self.norm = nn.LayerNorm(16); self.head = nn.Linear(16, 50, bias=False)

    def forward(self, x):
        return self.head(self.norm(self.fc(self.embed(x) + self.bucket(x % 40))))


def build(m, nd):
    return build_muon_adamw(m, muon_lr=0.005, adamw_lr=6e-4, weight_decay=0.1, adamw_fused=False,
                            route_embeddings_by_type=True, head_weight_decay=1.0, normuon=True,
                            fix_gate_matrices=True, no_decay_names=nd)


torch.manual_seed(0); m = Toy(); x = torch.randint(0, 50, (8, 12)); y = torch.randint(0, 50, (8, 12))
mu, ad, _, _ = build(m, None)
g = ad.param_groups
assert [len(q["params"]) for q in g] == [2, 3, 1] and g[0]["weight_decay"] == 0.1 and g[1]["weight_decay"] == 0.0 \
    and g[2]["weight_decay"] == 1.0, [(len(q["params"]), q["weight_decay"]) for q in g]
assert any(p is m.embed.weight for p in g[0]["params"])
for _ in range(5):
    loss = nn.functional.cross_entropy(m(x).reshape(-1, 50), y.reshape(-1)); loss.backward()
    mu.step(); ad.step(); mu.zero_grad(); ad.zero_grad()
saved = copy.deepcopy(ad.state_dict()); old_params = [p for q in ad.param_groups for p in q["params"]]
want = {id(p): saved["state"][i] for i, p in enumerate(old_params)}

mu2, ad2, _, _ = build(m, ["embed.weight"])
g2 = ad2.param_groups
assert [len(q["params"]) for q in g2] == [1, 4, 1], [len(q["params"]) for q in g2]
assert any(p is m.embed.weight for p in g2[1]["params"]) and any(p is m.bucket.weight for p in g2[0]["params"])
ad2.load_state_dict(remap_adamw_state(ad, ad2, saved))
for i, p in enumerate(p for q in ad2.param_groups for p in q["params"]):
    st = ad2.state_dict()["state"][i]
    assert torch.equal(st["exp_avg"], want[id(p)]["exp_avg"]) and torch.equal(st["exp_avg_sq"], want[id(p)]["exp_avg_sq"]) \
        and st["step"] == want[id(p)]["step"]
assert [q["weight_decay"] for q in ad2.param_groups] == [0.1, 0.0, 1.0]
e0, b0 = m.embed.weight.norm().item(), m.bucket.weight.norm().item()
for _ in range(300):                                        # zero gradient: only the decay acts
    for p in m.parameters(): p.grad = torch.zeros_like(p)
    ad2.step()
e1, b1 = m.embed.weight.norm().item(), m.bucket.weight.norm().item()
assert abs(e1 / e0 - 1) < 1e-3 and b1 / b0 < 0.99, (e1 / e0, b1 / b0)   # embed: only the leftover momentum moves it
print(f"[OK] grouping, remap (moments follow their tensor), decay: embed x{e1 / e0:.6f}, decayed table x{b1 / b0:.4f}")
