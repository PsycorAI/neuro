# temp/n85_write_cost.py -- 2026-10-28 review: per-write cost (remember + incremental _compile) at N existing memories,
# demo BUILD config, random unit embeddings + short random texts. CPU only.
import sys, time, random
import numpy as np
sys.path.insert(0, '/home/glenn/projects/neuro/psymem')
import core

rng, R = np.random.default_rng(0), random.Random(0)
words = [f'w{i}' for i in range(20000)]
p = core.Psymem(graded=1, signed=1, n_fly=65536, fly_k=4096)
for N in (1000, 5000, 10000):
    while len(p.diary) < N:
        e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
        p.remember('Me: ' + ' '.join(R.choices(words, k=15)), e, '2026-10-01', '2026-10-01', 'Me')
    t = time.perf_counter(); p._compile(); full = time.perf_counter() - t
    ts = []
    for _ in range(3):
        e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
        t = time.perf_counter()
        p.remember('Me: ' + ' '.join(R.choices(words, k=15)), e, '2026-10-27', '2026-10-27', 'Me'); p._compile()
        ts.append(time.perf_counter() - t)
    print(f'N {N:6d}: first compile {full:6.2f} s | one more write + incremental compile {1e3 * np.median(ts):8.1f} ms', flush=True)
