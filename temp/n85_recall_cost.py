# temp/n85_recall_cost.py -- 2026-10-28: recall latency, previous core (temp/n85_core_old.py) vs block-storage core, at
# 10k memories (demo config, random embeddings + texts). Memories added one batch at a time so the new core has
# frozen blocks + a live tail, as in real use.
import sys, time, random, importlib.util
import numpy as np
sys.path.insert(0, '/home/glenn/projects/neuro/psymem')
import core as new
spec = importlib.util.spec_from_file_location('core_old', '/home/glenn/projects/neuro/temp/n85_core_old.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
ACT = dict(g_sem=1, qrate=1, g_t=1, nb=0.5, nb_mode=1, nb_len=800, g_use=1, g_who=0.5)
R, rng = random.Random(0), np.random.default_rng(0)
words = [f'w{i}' for i in range(20000)]
for name, mod in (('old', old), ('new', new)):
    p = mod.Psymem(graded=1, signed=1, n_fly=65536, fly_k=4096)
    R.seed(0); rng = np.random.default_rng(0)
    for i in range(10100):
        e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
        p.remember('Me: ' + ' '.join(R.choices(words, k=15)), e, '2026-10-01', f's{i // 7}', 'Me')
        if i in (5000, 9000, 10000):
            p._compile()
    ts = []
    for _ in range(15):
        e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
        t = time.perf_counter(); p.activate(' '.join(R.choices(words, k=5)) + ' last week', e, now='2026-10-27', **ACT)
        ts.append(time.perf_counter() - t)
    print(f'{name}: recall median {1e3 * np.median(ts):.1f} ms  p90 {1e3 * np.percentile(ts, 90):.1f} ms', flush=True)
