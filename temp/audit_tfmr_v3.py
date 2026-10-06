import yaml
a = yaml.safe_load(open("/home/glenn/projects/neuro/configs/baseline_tfmr426_L17.yaml"))
b = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v3_363m.yaml"))
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
    if va != vb: print(f"{k:32s} tfmr={va!s:30.30s} v3={vb!s:.40s}")
