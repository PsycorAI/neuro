import sys, yaml, torch
sys.path.insert(0, '/home/glenn/projects/neuro/src')
from train_gpu import build_model
for n in sys.argv[1:]:
    c = yaml.safe_load(open('/home/glenn/projects/neuro/configs/' + n + '.yaml'))
    m = build_model(c)
    print(f'{n:45s} {sum(p.numel() for p in m.parameters())/1e6:7.2f}M')
