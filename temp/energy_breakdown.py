"""energy_breakdown.py -- 2026-09-30: where the per-token inference energy goes (src/energy.py formulas, 45nm MAC 4.6 pJ /
AC 0.9 pJ), with the TRUE firing rate. Spiking model = signed+canon 2L [spk, attn] (d=d_mem=1024, N=2048, V=16384).
Attention: full window vs a 64-token sliding window (~1% ppl cost measured, temp/eval_attn_window.py).
Baselines: 2L transformer (68.5 ppl) and the QUALITY-MATCHED 3L transformer (59.38 vs our 59.74). Pure arithmetic."""
E_MAC, E_AC = 4.6e-12, 0.9e-12
d = dm = 1024; N = 2048; V = 16384
def ours(T, rate, win=None):
    Te = T if win is None else min(T, win)
    parts = {"attn block dense (12d^2)": 12 * dm * dm * E_MAC, "attn scores (2dT)": 2 * dm * Te * E_MAC,
             "spk to_current (dN)": d * N * E_MAC, "GDN memory (3dm^2)": 3 * dm * dm * E_MAC,
             "spikes (7 proj, AC)": rate * 7 * N * dm * E_AC, "LM head (dV)": dm * V * E_MAC}
    return parts
def tfmr(L, T): return (L * (12 * d * d + 2 * d * T) + d * V) * E_MAC
for T in (512, 4096, 32768, 131072):
    p = ours(T, 0.125); tot = sum(p.values()); ps = ours(T, 0.125, 64); tots = sum(ps.values())
    print(f"T={T:6d}: ours {tot*1e6:6.1f} uJ (SWA64 {tots*1e6:6.1f}) | tfmr2L {tfmr(2,T)*1e6:7.1f} ({tfmr(2,T)/tot:4.2f}x, "
          f"SWA {tfmr(2,T)/tots:5.2f}x) | tfmr3L quality-matched {tfmr(3,T)*1e6:7.1f} ({tfmr(3,T)/tot:4.2f}x, SWA {tfmr(3,T)/tots:5.2f}x)")
p = ours(512, 0.125); tot = sum(p.values())
print("share @512: " + " | ".join(f"{k} {100*v/tot:.1f}%" for k, v in p.items()))
for r in (0.125, 0.10, 0.05):
    print(f"firing {r:.3f}: total @512 {sum(ours(512, r).values())*1e6:.2f} uJ")
