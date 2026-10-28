# n85_scale.py -- 2026-10-27 night (Neuro 85): Plasmem cost at scale on CPU: write time per memory (synaptic update only;
# the embedding is a separate front-end cost), consolidation (_compile) time, recall latency, synapse count and RAM.
# Texts = SODA turns (real word statistics); embeddings = random unit vectors (timing does not depend on their content).
# Shared config: big signed graded FlyHash (4,096 of 65,536) + time cells + gated neighbour links.
import sys, time, random, resource
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '/home/glenn/projects/neuro/psymem')
import core
BUILD = dict(graded=1, signed=1, n_fly=65536, fly_k=4096)
ACT = dict(g_sem=1, qrate=1, g_t=1, nb=0.5, nb_mode=1, nb_len=800)
t = pq.read_table('/home/glenn/projects/neuro/data/soda/train0.parquet', columns=['dialogue', 'speakers']).slice(0, 20000).to_pylist()
turns = [(f'{s}: {x}', d) for d, row in enumerate(t) for s, x in zip(row['speakers'], row['dialogue'])]
rng = np.random.default_rng(0)
for N in [int(a) for a in sys.argv[1:]] or [1000, 10000, 30000]:
    p = core.Psymem(**BUILD)
    E = rng.standard_normal((N, 384)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    t0 = time.perf_counter()
    for i in range(N):
        text, d = turns[i]
        p.remember(text, E[i], f'2026-{1 + d % 12:02d}-{1 + d % 28:02d}', d, text.split(':')[0])
    tw = time.perf_counter() - t0
    t0 = time.perf_counter(); p._compile(); tc = time.perf_counter() - t0
    qs = [turns[random.Random(i).randrange(N)][0].split(': ', 1)[1][:80] for i in range(30)]
    lat = []
    for q in qs:
        e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
        t0 = time.perf_counter(); p.activate(q, e, now='2026-12-31', **ACT); lat.append(time.perf_counter() - t0)
    syn = p.Wl.nnz + p.Ws.nnz
    mb = (p.Wl.data.nbytes + p.Wl.indices.nbytes + p.Ws.data.nbytes + p.Ws.indices.nbytes) / 1e6
    print(f'N {N:>7,}: write {1e6 * tw / N:7.1f} us/memory | consolidate {tc:6.2f} s | recall median {1e3 * np.median(lat):7.1f} ms '
          f'p90 {1e3 * np.percentile(lat, 90):7.1f} ms | synapses {syn / 1e6:7.1f} M ({mb:7.1f} MB) | '
          f'peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:5.2f} GB', flush=True)
