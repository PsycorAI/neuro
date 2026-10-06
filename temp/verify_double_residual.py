"""Verify: HybridAttentionBlock.forward_sequence already returns x + attn + ffn,
and SpikingHebbianLM.forward adds x_seq again -> stream = 2x + attn + ffn.
Tiny CPU model, attention at layer 1. Compare the stream after layer 1 with
block_out (which already contains x) and with block_out + x."""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from model import SpikingHebbianLM

torch.manual_seed(0)
m = SpikingHebbianLM(256, d=64, n_neurons=128, d_mem=64, n_layers=2, use_fpt=True,
                     delta_rule=True, gdn2_enabled=True, use_rmsnorm=True,
                     attention_layer_positions=[1], attention_heads=4, compile_safe=True).eval()
cap = {}
attn = m.blocks[1]
orig = attn.forward_sequence
def spy(x_seq, *a, **k):
    out = orig(x_seq, *a, **k)
    cap["x_in"], cap["blk_out"] = x_seq.detach(), out[0].detach()
    return out
attn.forward_sequence = spy
h = m.head
m.head = torch.nn.Identity()          # logits := final residual stream (no final_norm)
with torch.no_grad():
    stream = m(torch.randint(0, 256, (2, 8)))
m.head = h
x, bo = cap["x_in"], cap["blk_out"]
inner = bo - x                         # attn + ffn contribution
print("block already includes residual   :", torch.allclose(bo, x + inner))
print("stream == blk_out (single residual):", torch.allclose(stream, bo, atol=1e-5))
print("stream == blk_out + x (DOUBLE)     :", torch.allclose(stream, bo + x, atol=1e-5))
print(f"|x|={x.norm():.3f}  |attn+ffn|={inner.norm():.3f}  |stream|={stream.norm():.3f}")
