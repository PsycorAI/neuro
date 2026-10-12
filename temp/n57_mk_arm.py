"""n57_mk_arm.py -- 2026-10-12 (Neuro 57). Make a 13L SFT rehearsal arm from configs/sft_v4_reh13.yaml.
  python temp/n57_mk_arm.py <new_run_name> key=value [key=value ...]     (values parsed as YAML)"""
import sys, yaml
R = "/home/glenn/projects/neuro/configs/"
name = sys.argv[1]
c = yaml.safe_load(open(R + "sft_v4_reh13.yaml"))
upd = {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
c.update(upd); c["run_name"] = name
with open(R + name + ".yaml", "w") as f:
    f.write(f"# {name} -- AUTO-GEN temp/n57_mk_arm.py (2026-10-12) from sft_v4_reh13 + {upd}\n")
    yaml.safe_dump(c, f, sort_keys=False)
print(f"[OK] {name}.yaml:", upd)
