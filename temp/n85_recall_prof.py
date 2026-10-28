# temp/n85_recall_prof.py -- 2026-10-28: where the block-storage recall spends its time at 10k memories.
import sys, time, random, cProfile, pstats
import numpy as np
sys.path.insert(0, '/home/glenn/projects/neuro/psymem')
import core
ACT = dict(g_sem=1, qrate=1, g_t=1, nb=0.5, nb_mode=1, nb_len=800, g_use=1, g_who=0.5)
R, rng = random.Random(0), np.random.default_rng(0)
words = [f'w{i}' for i in range(20000)]
p = core.Psymem(graded=1, signed=1, n_fly=65536, fly_k=4096)
for i in range(10100):
    e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
    p.remember('Me: ' + ' '.join(R.choices(words, k=15)), e, '2026-10-01', f's{i // 7}', 'Me')
    if i in (5000, 9000, 10000):
        p._compile()
p._compile()
print('blocks', [b.shape[1] for b in p._sb], 'tail', len(p._tc))
qs = []
for _ in range(15):
    e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
    qs.append((' '.join(R.choices(words, k=5)) + ' last week', e))
pr = cProfile.Profile(); pr.enable()
for q, e in qs:
    p.activate(q, e, now='2026-10-27', **ACT)
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(12)
