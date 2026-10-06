"""r16_init_scales.py -- 2026-10-02. At init, on the real 30M C0 shape (CPU): RMS of the residual stream entering
block 0, of the H block's increment (with and without its output LayerNorm), and of each attention block's
increment. Used to pick spk_layerscale for spk_out_norm: false."""
import sys, torch, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)
from train_gpu import build_model
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_r14b_3Lattn2_canon_ramp10_s0_30m.yaml"))
c.update(n_layers=4, attention_layer_positions=[1, 2, 3], engram_v2_hash="hi", lif_levels=1,
         attn_ffn_spikeglu_k=256, attn_ffn_hidden=2752)
torch.manual_seed(0); X = torch.randint(0, 16384, (1, 128))
for out_norm in (True, False):
    torch.manual_seed(0); m = build_model(dict(c, spk_out_norm=out_norm)); m.eval(); rec = {}
    def hook(i):
        def f(mod, inp, out):
            rec[i] = (inp[0].float().pow(2).mean().sqrt().item(), out[0].float().pow(2).mean().sqrt().item(),
                      (out[0] - inp[0]).float().pow(2).mean().sqrt().item())
        return f
    for i, b in enumerate(m.blocks):
        def wrap(fs, i):
            def g(x_seq, *a, **k):
                r = fs(x_seq, *a, **k)
                rec[i] = (x_seq.float().pow(2).mean().sqrt().item(), r[0].float().pow(2).mean().sqrt().item())
                return r
            return g
        b.forward_sequence = wrap(b.forward_sequence, i)
    with torch.no_grad():
        m(X)
    print(f"spk_out_norm={out_norm}: " + " | ".join(
        f"blk{i} in-RMS {a:.3f} out-RMS {o:.3f}" for i, (a, o) in sorted(rec.items())))
print("(H block 0 returns its increment (the LM loop adds the residual); attention blocks return x + increment)")
