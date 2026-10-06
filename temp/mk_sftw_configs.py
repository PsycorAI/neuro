# mk_sftw_configs.py -- 2026-09-26 round 3 of the SFT MQAR-rehearsal sweep (Glenn: find the
# best setting between w0.2 and w0.5, more variations welcome). All from screen_sftab_mixmqar
# (v3 mix, 6M tok, from the 10B base); ONE change each:
#   w01   : mqar_weight 0.1
#   w03   : mqar_weight 0.3
#   hin02 : mqar_weight 0.2, rehearse only N in [16,32] (where SFT loses recall; N<=16 holds)
#   w02s1 : mqar_weight 0.2, seed 1 (training-noise estimate vs mixmqar02 = seed 0)
import re, pathlib
base = pathlib.Path("configs/screen_sftab_mixmqar_363m.yaml").read_text()
def setk(s, k, v):
    s2, n = re.subn(rf"^{k}: .*$", f"{k}: {v}", s, flags=re.M); assert n == 1, k; return s2
arms = {
    "w01":   {"mqar_weight": 0.1},
    "w03":   {"mqar_weight": 0.3},
    "hin02": {"mqar_weight": 0.2, "mqar_N_min": 16},
    "w02s1": {"mqar_weight": 0.2, "seed": 1},
}
for tag, kv in arms.items():
    s = setk(base, "run_name", f"screen_sftab_{tag}_363m")
    for k, v in kv.items():
        s = setk(s, k, v)
    p = pathlib.Path(f"configs/screen_sftab_{tag}_363m.yaml")
    p.write_text(s)
    print("wrote", p, kv)
