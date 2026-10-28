# n85_prof.py -- 2026-10-28: profile one Psymem recall at 10k spans (where does the time go?).
import sys, cProfile, pstats, numpy as np
sys.path.insert(0, '/home/glenn/projects/neuro/psymem'); sys.argv = ['x', '10000']
exec(open('/home/glenn/projects/neuro/temp/n85_scale.py').read().split('for N in')[0])
N = 10000
p = core.Psymem(**BUILD)
E = rng.standard_normal((N, 384)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
for i in range(N):
    text, d = turns[i]; p.remember(text, E[i], f'2026-{1 + d % 12:02d}-{1 + d % 28:02d}', d, text.split(':')[0])
p._compile()
e = E[5]
cProfile.run("for _ in range(10): p.activate('what did she say about the bakery last week', e, now='2026-12-31', **ACT)", '/tmp/prof')
pstats.Stats('/tmp/prof').sort_stats('cumulative').print_stats(18)
