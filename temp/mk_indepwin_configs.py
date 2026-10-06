"""Generate the 30M independent-windows screen configs (2026-09-25).
Control (reused): screen_bs512_s{0,1}_30m. ONE change: data_independent_windows: true."""
import pathlib
for s in (0, 1):
    src = pathlib.Path(f"configs/screen_bs512_s{s}_30m.yaml").read_text().splitlines()
    body = [l for l in src if not l.lstrip().startswith("#")]
    body = [f"run_name: screen_indepwin_s{s}_30m" if l.startswith("run_name:") else l for l in body]
    body.append("data_independent_windows: true")
    hdr = (f"# screen_indepwin_s{s}_30m -- data-sampling screen (2026-09-25). AUTO-GEN by\n"
           f"# temp/mk_indepwin_configs.py from screen_bs512_s{s}_30m (the control). ONE change:\n"
           f"# each train window gets its own random start (default draws all B windows from\n"
           f"# one ~4*B*(T+1)-token region -> correlated microbatches). Val sampler unchanged.\n")
    p = pathlib.Path(f"configs/screen_indepwin_s{s}_30m.yaml")
    p.write_text(hdr + "\n".join(body) + "\n"); print("wrote", p)
