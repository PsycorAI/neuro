"""mk_r19_seeds.py -- 2026-10-03: seed repeats of r19 winners (identical except seed/run_name)."""
import yaml, sys
R = '/home/glenn/projects/neuro/configs/'
for arm in sys.argv[1:]:
    for s in (1, 2):
        c = yaml.safe_load(open(R + f'screen_{arm}_s0_30m.yaml')); name = f'screen_{arm}_s{s}_30m'
        c.update(seed=s, run_name=name)
        with open(R + name + '.yaml', 'w') as f:
            f.write(f'# {name} -- AUTO-GEN temp/mk_r19_seeds.py (2026-10-03) from screen_{arm}_s0_30m. Seed {s} repeat.\n')
            yaml.safe_dump(c, f, sort_keys=False)
        print('[OK]', name)
