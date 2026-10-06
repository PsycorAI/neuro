"""check_insert.py -- 2026-10-05: the remapped r25l seed must reproduce r21b's logits up to the near-silent new block
(spk_layerscale 0.01), and load with only the new block's params missing. CPU, short input."""
import sys, torch, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
src = torch.load("/home/glenn/projects/neuro/models/checkpoints/r21b_x1_150m/step_0012208.pt", map_location="cpu",
                 weights_only=False)
new = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m0 = build_model(dict(src["cfg"])).cuda(); m0.load_state_dict(src["model"], strict=False); m0.eval()
m1 = build_model(dict(new["cfg"])).cuda(); miss = m1.load_state_dict(new["model"], strict=False); m1.eval()
print("missing:", len(miss.missing_keys), [k for k in miss.missing_keys if not k.startswith("blocks.6.")][:5],
      "| unexpected:", len(miss.unexpected_keys))
x = torch.randint(0, 32768, (1, 128)).cuda()
with torch.no_grad():
    a, b = m0(x).float(), m1(x).float()
p, q = torch.softmax(a, -1), torch.softmax(b, -1)
print(f"max|dlogit| {(a - b).abs().max().item():.3f} | mean TV {0.5 * (p - q).abs().sum(-1).mean().item():.4f} | "
      f"top1 agree {(a.argmax(-1) == b.argmax(-1)).float().mean().item():.3f}")
