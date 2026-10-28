# scripts/memdemo/engine.py -- 2026-10-27 (Neuro 85): the Plasmem demo engine. Wraps psymem.core.Psymem (the shared
# benchmark config: big signed graded FlyHash + time cells + neighbour links) and adds explain(): what fired, what each
# channel gave, confidence, and plain vector search (same embedder, cosine) for comparison. CPU only.
# The memory lives in one file, logs/memdemo/<name>.plasmem (npz: embeddings + JSON of diary, word neurons, use synapses).
import os, sys, json, time, datetime
import numpy as np
R = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../..') + '/'
sys.path.insert(0, R + 'psymem')
import core, bench

BUILD = dict(graded=1, signed=1, n_fly=65536, fly_k=4096)
ACT = dict(g_sem=1, qrate=1, g_t=1, nb=0.5, g_use=1)
G_WHO = 0.5  # who-cell gate when on (not in the benchmark config; a demo toggle)
SEED = os.path.dirname(os.path.abspath(__file__)) + '/diary.json'


class Embedder(bench.Embedder):  # bge-small, fp32 on CPU (same model and pooling as the benchmarks)
    def __init__(self):
        super().__init__(dev='cpu')
        self.m = self.m.float()


class Engine:
    def __init__(self, name='diary', emb=None):
        self.file = R + f'logs/memdemo/{name}.plasmem'
        self.emb = emb or Embedder()
        seed = json.load(open(SEED))
        self.today = seed['today']
        self.load() if os.path.exists(self.file) else self.reset()

    # ---- memory file ----
    def reset(self):
        seed = json.load(open(SEED))
        self.p = core.Psymem(**BUILD)
        items = [(f'{w}: {t}', s['date'], w) for s in seed['sessions'] for w, t in s['turns']]
        for (text, date, who), e in zip(items, self.emb([i[0] for i in items])):
            self.p.remember(text, e, date, date, who)
        self.save()

    def save(self):
        p = self.p
        meta = dict(diary=p.diary, lex=p.lex, use_l=[[c, s, v] for (c, s), v in p.use_l.items()],
                    use_s=[[int(j), s, v] for (j, s), v in p.use_s.items()], build=BUILD)
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file + '.tmp', 'wb') as f:
            np.savez_compressed(f, E=np.stack(p.E).astype(np.float16), meta=np.array(json.dumps(meta)))
        os.replace(self.file + '.tmp', self.file)

    def load(self):
        z = np.load(self.file)
        meta = json.loads(str(z['meta']))
        self.p = p = core.Psymem(**meta['build'])
        p.lex = meta['lex']  # word neurons keep their ids (some were grown by learning between writes)
        for (text, date, sid, who), e in zip(meta['diary'], z['E'].astype(np.float32)):
            p.remember(text, e, date, sid, who)
        p.use_l = {(c, s): v for c, s, v in meta['use_l']}
        p.use_s = {(j, s): v for j, s, v in meta['use_s']}
        p.dirty = True

    def remember(self, text, who, date):
        self.p.remember(f'{who}: {text}', self.emb([f'{who}: {text}'])[0], date, date, who)
        self.save()
        return len(self.p.diary) - 1

    def diary(self):
        return [dict(i=i, who=w, date=d, text=t.split(': ', 1)[-1]) for i, (t, d, s, w) in enumerate(self.p.diary)]

    def stats(self):
        p = self.p
        if p.dirty:
            p._compile()
        syn = p.Wl.nnz + p.Ws.nnz + len(p.use_l) + len(p.use_s)
        return dict(memories=len(p.diary), synapses=syn, use_synapses=len(p.use_l) + len(p.use_s),
                    bytes=os.path.getsize(self.file) if os.path.exists(self.file) else 0, word_neurons=len(p.lex),
                    meaning_neurons=len(p.P))

    # ---- recall ----
    def qemb(self, q):
        return self.emb([bench.Embedder.QPFX + q])[0]

    def rank(self, q, span, now=None, who=True):  # 1-based rank of one memory for a question
        a = self.p.activate(q, self.qemb(q), now=now or self.today, g_who=G_WHO if who else 0, **ACT)
        return int((a > a[span]).sum()) + 1

    def ask(self, q, now=None, who=True, k=5, gold=None):
        t0, p, now = time.perf_counter(), self.p, now or self.today
        e, g_who = self.qemb(q), (G_WHO if who else 0)
        a = p.activate(q, e, now=now, g_who=g_who, **ACT)
        conf = p.confidence(q, e, g_sem=ACT['g_sem'], g_who=g_who)
        ms = (time.perf_counter() - t0) * 1e3
        # channel split, recomputed exactly as Psymem.activate does it (prf and adapt are off in this config)
        words = [w for w in core.bag(q) if w in p.lex]
        ids, code = [p.lex[w] for w in words], p._code(e[None])[0]
        Ml, Ms = p.Wl[ids].toarray(), p.Ws[code].toarray()
        h = (e - p.mu) @ p.P[code].T
        gs = (h - h.min() + 1e-6) / (h - h.min() + 1e-6).mean()
        a_s0 = gs @ Ms; med = np.median(a_s0)
        lex = Ml.sum(0) / (Ml.sum(0).max() or 1.)
        sem = ACT['g_sem'] * np.maximum(a_s0 - med, 0) / ((a_s0.max() - med) or 1.)
        use = np.zeros_like(lex)
        if p.use_l or p.use_s:
            idf = p.idf[ids]
            use = ACT['g_use'] * (idf @ p.Ul[ids].toarray() / (idf.sum() or 1.)
                                  + ACT['g_sem'] * np.asarray(p.Us[code].sum(0)).ravel() / p.fly_k)
        t_now, win = core.day(now), None
        tgate = np.ones_like(lex)
        if w := core.when(q, t_now):
            win, tgate = w, 1 + ACT['g_t'] * p._time(*w, t_now)
        wgate = p._who(q, g_who) * np.ones_like(lex)
        base = (lex + sem + use) * tgate * wgate
        nbr = np.maximum(a - base, 0)
        top = np.argsort(-a)[:k]
        vec = np.stack(p.E) @ e
        named = [w for w in p.whos if w in core.bag(q)] if g_who else []
        out = []
        for s in top:
            parts = dict(word=lex[s] * tgate[s] * wgate[s], meaning=sem[s] * tgate[s] * wgate[s],
                         learned=use[s] * tgate[s] * wgate[s], neighbour=nbr[s])
            t, d, sid, who_s = p.diary[s]
            out.append(dict(i=int(s), score=float(a[s]), who=who_s, date=d, text=t.split(': ', 1)[-1],
                            parts={k2: float(v) for k2, v in parts.items()}, time=float(p._time(*win, t_now)[s]) if win else None,
                            words=[w for w, r in zip(words, Ml[:, s]) if r > 0],
                            meaning_overlap=int((Ms[:, s] > 0).sum())))
        cue = [dict(w=w, idf=float(p.idf[p.lex[w]]), fan=int(p.Wl[p.lex[w]].nnz)) for w in words]
        used = int((Ml > 0).sum() + (Ms > 0).sum())
        st = self.stats()
        gr = dict(gold=gold, gold_rank=int((a > a[gold]).sum()) + 1, gold_vec_rank=int((vec > vec[gold]).sum()) + 1) if gold is not None else {}
        return dict(q=q, now=now, confidence=conf, hits=out, **gr,
                    vector=[dict(i=int(s), score=float(vec[s]), who=p.diary[s][3], date=p.diary[s][1],
                                 text=p.diary[s][0].split(': ', 1)[-1]) for s in np.argsort(-vec)[:k]],
                    cue=cue, unknown=[w for w in core.bag(q) if w not in p.lex], meaning_fired=int(p.fly_k),
                    meaning_total=int(len(p.P)), synapses_used=used, synapses_total=st['synapses'], ms=ms,
                    time_window=[datetime.date.fromordinal(x).isoformat() for x in win] if win else None, who=named,
                    days=[int(x) for x in p.days], today_day=t_now)

    def feedback(self, q, span, now=None, who=True, probes=()):
        """Learn with use: the user confirms memory `span` answers q. Returns its rank before / after, for q and probes."""
        before = [self.rank(x, span, now, who) for x in (q, *probes)]
        n0 = len(self.p.use_l) + len(self.p.use_s)
        self.p.reinforce(q, self.qemb(q), [span])
        after = [self.rank(x, span, now, who) for x in (q, *probes)]
        self.save()
        return dict(before=before, after=after, grown=len(self.p.use_l) + len(self.p.use_s) - n0)
