#!/usr/bin/env python3
# mk_probe_configs.py -- generate v4 width-probe configs from the v4 draft.
# Two flavors per width W in {4480, 5120, 5760} (= 3.5, 4.0, 4.5 x d=1280),
# all holding gate_rank:16 (the adopted v4 lever):
#   temp/probe_b1_wW.yaml   : batch1, compile off, ~a few steps -> exact param
#                             count + batch-1 activation VRAM, runnable on the 5080.
#   temp/probe_b18_wW.yaml  : batch18, compiled (real recipe) -> the DEFINITIVE
#                             5090 VRAM/WDDM-cliff probe, to run when the 5090 frees.
import re, pathlib
src = pathlib.Path("configs/flagship_v4_363m.draft.yaml").read_text().splitlines()

# strip the leading comment banner (lines starting with '#') so overrides are clean
body = [l for l in src if not l.lstrip().startswith("#")]

def set_kv(lines, key, val):
    out, hit = [], False
    for l in lines:
        m = re.match(rf"^(\s*){re.escape(key)}\s*:", l)
        if m and not hit:
            out.append(f"{m.group(1)}{key}: {val}"); hit = True
        else:
            out.append(l)
    if not hit:
        out.append(f"{key}: {val}")
    return out

WIDTHS = [3840, 4480, 5120, 5760]   # 3840 = current flagship width (baseline anchor)
for W in WIDTHS:
    for tag, over in [
        ("b1",  dict(batch_size=1,  grad_accum=1, compile="false",
                     max_tokens=40000,  eval_every=100000, ckpt_every=100000, keep_last=1)),
        ("b18", dict(batch_size=18, grad_accum=4, compile="true",
                     max_tokens=3000000, eval_every=100000, ckpt_every=100000, keep_last=1)),
    ]:
        L = list(body)
        L = set_kv(L, "n_neurons", W)
        L = set_kv(L, "run_name", f"probe_{tag}_w{W}")
        for k, v in over.items():
            L = set_kv(L, k, v)
        hdr = (f"# probe_{tag}_w{W} -- v4 width VRAM/param probe (n_neurons {W} = "
               f"{W/1280:.1f}xd, gate_rank:16). AUTO-GEN by temp/mk_probe_configs.py. "
               f"{'batch1/compile-off: params + batch-1 VRAM (5080-safe).' if tag=='b1' else 'batch18/compiled: DEFINITIVE 5090 VRAM probe.'}")
        pathlib.Path(f"temp/probe_{tag}_w{W}.yaml").write_text(hdr + "\n" + "\n".join(L) + "\n")
        print(f"wrote temp/probe_{tag}_w{W}.yaml")
