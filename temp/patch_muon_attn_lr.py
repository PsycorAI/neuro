"""patch_muon_attn_lr.py -- 2026-09-28: optional separate Muon LR for HybridAttentionBlock matrices (qkv, o_proj, ffn)
(config muon_attn_lr, default None = byte-identical grouping). Why: the hybrid [spk, attn] ran its attention layer at the
spiking muon_lr 0.005; the 30M transformer's Muon sweep peaked at 0.02 (and NorMuon 0.005 cost it 87.8 vs 68.5).
  python temp/patch_muon_attn_lr.py <out_dir>"""
import sys, os
S = "/home/glenn/projects/neuro/src/"; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def rep(s, a, b):
    assert s.count(a) == 1, (a[:70], s.count(a)); return s.replace(a, b)
m = open(S + "muon.py", encoding="utf-8").read()
m = rep(m, "head_weight_decay=None, muon_mem_lr=None, batched_ns=False):",
        "head_weight_decay=None, muon_mem_lr=None, batched_ns=False,\n                    muon_attn_lr=None):")
m = rep(m, "    if not muon_params:\n        muon = None\n    elif muon_mem_lr is not None:",
        """    if not muon_params:
        muon = None
    elif muon_attn_lr is not None:
        # Separate LR for the hybrid attention block's matrices (qkv / o_proj / ffn -- names only the
        # HybridAttentionBlock has). Same proportional-schedule mechanism as muon_mem_lr. Extra param
        # group -> NOT resume-compatible; fresh runs only. Mutually exclusive with muon_mem_lr.
        if muon_mem_lr is not None:
            raise ValueError("muon_attn_lr and muon_mem_lr are mutually exclusive")
        attn_ids = {id(p) for n, p in model.named_parameters()
                    if any(s in n for s in (".qkv.", ".o_proj.", ".ffn."))}
        a_muon = [p for p in muon_params if id(p) in attn_ids]
        b_muon = [p for p in muon_params if id(p) not in attn_ids]
        groups = [g for g in ({"params": b_muon, "lr": muon_lr}, {"params": a_muon, "lr": muon_attn_lr}) if g["params"]]
        print(f"  muon_attn_lr={muon_attn_lr}: {len(a_muon)} attention matrices, {len(b_muon)} others at {muon_lr}")
        muon = Muon(groups, lr=muon_lr, weight_decay=weight_decay,
                    normuon=normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
                    batched_ns=batched_ns)
    elif muon_mem_lr is not None:""")
open(os.path.join(out, "muon.py"), "w", encoding="utf-8", newline="").write(m)
t = open(S + "train_gpu.py", encoding="utf-8").read()
t = rep(t, '            muon_mem_lr=c.get("muon_mem_lr"),\n',
        '            muon_mem_lr=c.get("muon_mem_lr"),\n            muon_attn_lr=c.get("muon_attn_lr"),\n')
open(os.path.join(out, "train_gpu.py"), "w", encoding="utf-8", newline="").write(t)
print("[OK] ->", out)
