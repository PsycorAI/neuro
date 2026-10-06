import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_parallel as LP
from lif_exact import lif_exact
torch.manual_seed(0)
for dt in (torch.float32, torch.bfloat16):
    c0 = (torch.randn(4, 512, 1024, device="cuda") * 0.6 + 0.1).to(dt)
    a = c0.clone().requires_grad_(); b = c0.clone().requires_grad_()
    gs = torch.randn(4, 512, 1024, device="cuda"); gm = torch.randn(4, 1024, device="cuda")
    s1, m1 = LP.lif_parallel(a, 0.9, K=0); (s1.float() * gs).sum().add((m1.float() * gm).sum()).backward()
    s2, m2 = lif_exact(b, 0.9, gamma=0.0); (s2.float() * gs).sum().add((m2.float() * gm).sum()).backward()
    flips = (s1.detach().round() != s2.detach().round()).float().mean().item()
    ga, gb = a.grad.float(), b.grad.float()
    print(f"{dt}: spike flips {flips:.2e} | rate {s1.float().mean():.3f} vs {s2.float().mean():.3f} | "
          f"mem_final max diff {(m1.float()-m2.float()).abs().max():.2e} | grad rel err {((ga-gb).abs().max()/ga.abs().max()).item():.2e} | "
          f"grad norms {ga.norm():.3f} vs {gb.norm():.3f}")
