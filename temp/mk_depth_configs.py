"""Generate the 30M depth-vs-width screen configs (2026-09-25).
Control (reused, already run): screen_bs512_s{0,1}_30m = d1024 L2 N2048 (59.6M).
Treatment: L4 d832 N1664 (60.1M) -- same N/d=2.0, 2x depth, param-matched.
Question: at fixed params, does depth beat width? Informs v4 A' (9L wide) vs B (12L)."""
import re, pathlib
for s in (0, 1):
    src = pathlib.Path(f"configs/screen_bs512_s{s}_30m.yaml").read_text().splitlines()
    body = [l for l in src if not l.lstrip().startswith("#")]
    over = dict(d=832, d_mem=832, n_neurons=1664, n_layers=4, run_name=f"screen_depth_L4_s{s}_30m")
    out = []
    for l in body:
        m = re.match(r"^(\w+)\s*:", l)
        out.append(f"{m.group(1)}: {over[m.group(1)]}" if m and m.group(1) in over else l)
    hdr = (f"# screen_depth_L4_s{s}_30m -- depth-vs-width screen (2026-09-25). AUTO-GEN by\n"
           f"# temp/mk_depth_configs.py from screen_bs512_s{s}_30m (the control). ONE change:\n"
           f"# L2 d1024 N2048 (59.6M) -> L4 d832 N1664 (60.1M), same N/d=2.0, param-matched.\n"
           f"# CAVEAT: 30M has no attention layer. Informs v4 shape A' (9L wide) vs B (12L).\n")
    p = pathlib.Path(f"configs/screen_depth_L4_s{s}_30m.yaml")
    p.write_text(hdr + "\n".join(out) + "\n"); print("wrote", p)
