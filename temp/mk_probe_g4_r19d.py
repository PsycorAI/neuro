"""mk_probe_g4_r19d.py -- 2026-10-03: 1B G4 probe (probe_g4_c0s1) with the r19d sparse MLP instead of DSN:
relu2 + per-neuron homeostatic threshold -> 10%, hidden 12288 (2 matrices = param-matched to DSN 3x8192)."""
import yaml
R = '/home/glenn/projects/neuro/configs/'
c = yaml.safe_load(open(R + 'probe_g4_c0s1.yaml'))
for k in ('attn_ffn_spikeglu_k', 'attn_ffn_spikeglu_ramp_start', 'attn_ffn_spikeglu_ramp_end'):
    c.pop(k)
c.update(attn_ffn_hidden=12288, attn_ffn_act='relu2', attn_ffn_homeo='unit', attn_ffn_rate_target=0.1,
         attn_ffn_ramp_start=0.6, attn_ffn_ramp_end=0.9, attn_ffn_homeo_lr=0.02, run_name='probe_g4_r19d')
with open(R + 'probe_g4_r19d.yaml', 'w') as f:
    f.write('# probe_g4_r19d -- AUTO-GEN temp/mk_probe_g4_r19d.py (2026-10-03) from probe_g4_c0s1: DSN -> r19d sparse MLP.\n')
    yaml.safe_dump(c, f, sort_keys=False)
print('[OK] probe_g4_r19d')
