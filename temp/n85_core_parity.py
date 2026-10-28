# temp/n85_core_parity.py -- 2026-10-28: new block-storage core (psymem/core.py) vs the previous core
# (temp/n85_core_old.py = git HEAD) on the same write / reinforce / recall sequence. Checks activations and confidence
# are identical (max abs diff 0 or float noise) across configs, through block freezes, merges, a mean-drift re-derive.
import sys, random, importlib.util
import numpy as np
sys.path.insert(0, '/home/glenn/projects/neuro/psymem')
import core as new
spec = importlib.util.spec_from_file_location('core_old', '/home/glenn/projects/neuro/temp/n85_core_old.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

R, rng = random.Random(0), np.random.default_rng(0)
vocab = [f'w{i}' for i in range(3000)]
names = ['Maya', 'Sam', 'Lena', 'Omar']
BUILD = dict(graded=1, signed=1, n_fly=65536, fly_k=4096)
CFGS = [dict(g_sem=1, qrate=1, g_t=1, nb=0.5, nb_mode=1, nb_len=800, g_use=1, g_who=0.5),
        dict(g_sem=1, qrate=1, g_t=1, nb=0.5, nb_mode=0),
        dict(g_sem=0.7, nb=0.5, nb_mode=2), dict(g_sem=1, nb=0.5, nb_mode=3, nb_len=60),
        dict(g_sem=1, prf=10), dict(g_sem=1, adapt=0.3, K=5)]
a, b = old.Psymem(**BUILD), new.Psymem(**BUILD)
worst, checks = 0., 0


def write(i, shift):
    who = R.choice(names)
    text = f'{who}: ' + ' '.join(R.choices(vocab, k=R.randint(3, 40))) + (' ' + R.choice(names) if R.random() < .3 else '') \
        + ('?' if R.random() < .2 else '')
    e = rng.standard_normal(384).astype(np.float32) + shift; e /= np.linalg.norm(e)
    date = f'2026-{1 + i // 120 % 12:02d}-{1 + i % 28:02d}'
    for p in (a, b):
        p.remember(text, e, date, f's{i // 7}', who)


def check(tag):
    global worst, checks
    for _ in range(4):
        q = ' '.join(R.choices(vocab, k=4)) + R.choice(['', ' last week', ' in March', ' two weeks ago']) + \
            (' ' + R.choice(names) if R.random() < .5 else '') + (' zzunknown' if R.random() < .3 else '')
        e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
        for c in CFGS:
            x, y = a.activate(q, e, now='2026-10-27', **c), b.activate(q, e, now='2026-10-27', **c)
            d = float(np.abs(x - y).max()); worst = max(worst, d); checks += 1
            assert d < 1e-9 and (np.argsort(-x)[:10] == np.argsort(-y)[:10]).all(), (tag, c, d)
        d = abs(a.confidence(q, e, g_sem=1, g_who=0.5) - b.confidence(q, e, g_sem=1, g_who=0.5)); worst = max(worst, d)
        assert d < 1e-9, (tag, 'confidence', d)
    assert (a.Wl != b.Wl).nnz == 0 and abs(a.Ws - b.Ws).max() < 1e-12, tag
    assert (a.who_w == b.who_w).all() and a.whos == b.whos and (a.days == b.days).all() and (a.sid == b.sid).all(), tag


i = 0
for stop, every in [(300, 50), (330, 1), (900, 100), (1300, 37)]:  # batched + one-at-a-time compiles; 256-freeze, merges
    while i < stop:
        write(i, 0.); i += 1
        if i % every == 0:
            check(f'n={i}')
    q = ' '.join(R.choices(vocab, k=3)); e = rng.standard_normal(384).astype(np.float32); e /= np.linalg.norm(e)
    s = R.randrange(i)
    for p in (a, b):
        p.reinforce(q, e, [s])
    check(f'reinforce n={i}')
print('blocks', [m.shape[1] for m in b._sb], 'tail', len(b._tc))
for _ in range(400):  # shift the embedding distribution: the centring mean drifts > 5% -> full re-derive in both
    write(i, 0.15); i += 1
check(f'drift n={i}')
print(f'PARITY OK: {checks} activation checks, worst abs diff {worst:.2e}; blocks', [m.shape[1] for m in b._sb], 'tail', len(b._tc))
