# psymem/bench.py -- 2026-10-27 (Neuro 84): retrieval benchmark for the memory product (codename Psymem).
# A "memory" = one LoCoMo conversation or one LongMemEval haystack; items = dialogue turns. Each scorer gets the per-memory
# context and a question and returns one score per item. Metrics (per question category, micro-averaged):
#   r@k   = share of the gold evidence turns found in the top k;  all@k = every gold turn found (what multi-hop needs);
#   sr@k  = LongMemEval only: share of the answer sessions with at least one turn in the top k.
# Usage: python psymem/bench.py locomo|lme [systems...]      (abstention questions are left out here; tested separately)
import json, re, sys, math, time, collections, hashlib, pickle, os
import numpy as np, torch
import core
from core import bag

D ='/home/glenn/projects/neuro/data/membench/'
KS = (5, 10, 20)


def load_locomo():
    out = []
    for conv in json.load(open(D + 'locomo10.json')):
        c, items = conv['conversation'], []
        for k in c:
            if re.fullmatch(r'session_\d+', k):
                for t in c[k]:
                    cap = f" [shares a photo: {t['blip_caption']}]" if t.get('blip_caption') else ''
                    items.append((t['dia_id'], f"{t['speaker']}: {t['text']}{cap}", c[k + '_date_time'], k))
        ids = {i[0] for i in items}
        qs = [(q['question'], [e for e in q['evidence'] if e in ids], q['category'],
               str(q.get('answer', q.get('adversarial_answer'))), None) for q in conv['qa']]
        out.append(dict(items=items, qs=[q for q in qs if q[1]]))
    return out


def load_lme(name='longmemeval_s_cleaned.json', abs_too=False):
    out = []
    for x in json.load(open(D + name)):
        if x['question_id'].endswith('_abs') and not abs_too:
            continue
        items, gold = [], []
        for sid, date, sess in zip(x['haystack_session_ids'], x['haystack_dates'], x['haystack_sessions']):
            for j, t in enumerate(sess):
                items.append((f'{sid}:{j}', f"{t['role']}: {t['content']}", date, sid))
                if t.get('has_answer'):
                    gold.append(f'{sid}:{j}')
        qtype = 'abstain' if x['question_id'].endswith('_abs') else x['question_type']
        out.append(dict(items=items, qs=[(x['question'], gold, qtype, str(x['answer']), x['question_date'])],
                        ans=set(x['answer_session_ids'])))
    return out


# ---- ordinary memory search: BM25, dense embeddings (bge-small), hybrid (reciprocal rank fusion) ----
class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        docs = [bag(t) for t in texts]
        self.n, ln = len(docs), np.array([sum(d.values()) for d in docs], float)
        df = collections.Counter(w for d in docs for w in d)
        self.idf = {w: math.log(1 + (self.n - f + .5) / (f + .5)) for w, f in df.items()}
        self.inv = collections.defaultdict(list)
        norm = k1 * (1 - b + b * ln / ln.mean())
        for i, d in enumerate(docs):
            for w, f in d.items():
                self.inv[w].append((i, f * (k1 + 1) / (f + norm[i])))

    def scores(self, q):
        s = np.zeros(self.n)
        for w in bag(q):
            for i, v in self.inv.get(w, ()):
                s[i] += self.idf[w] * v
        return s


class Embedder:
    QPFX = 'Represent this sentence for searching relevant passages: '

    def __init__(self, path='BAAI/bge-small-en-v1.5', dev='cuda', cache_file=None):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path, dtype=torch.float16).to(dev).eval()
        self.dev, self.file = dev, cache_file  # disk cache: sha1(text)[:12] -> fp16 vector
        self.cache = pickle.load(open(cache_file, 'rb')) if cache_file and os.path.exists(cache_file) else {}
        self.n0 = len(self.cache)

    @staticmethod
    def key(t):
        return hashlib.sha1(t.encode()).digest()[:12]

    @torch.no_grad()
    def __call__(self, texts, bs=256):
        need = sorted({t for t in texts if self.key(t) not in self.cache}, key=len)
        for i in range(0, len(need), bs):
            b = self.tok(need[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.dev)
            e = torch.nn.functional.normalize(self.m(**b).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
            self.cache.update(zip(map(self.key, need[i:i + bs]), e.astype(np.float16)))
        return np.stack([self.cache[self.key(t)] for t in texts]).astype(np.float32)

    def save(self):
        if self.file and len(self.cache) > self.n0:
            pickle.dump(self.cache, open(self.file, 'wb'))


def ranks(s):
    r = np.empty(len(s)); r[np.argsort(-s)] = np.arange(len(s)); return r


def rrf(*scores, k=60):
    return sum(1 / (k + 1 + ranks(s)) for s in scores)


def ctx_base(m, emb):
    texts = [i[1] for i in m['items']]
    return dict(items=m['items'], bm25=BM25(texts), E=emb(texts), emb=emb)


SCORERS = {
    'bm25': lambda c, q: c['bm25'].scores(q),
    'dense': lambda c, q: c['E'] @ c['emb']([Embedder.QPFX + q])[0],
    'hybrid': lambda c, q: rrf(c['bm25'].scores(q), c['E'] @ c['emb']([Embedder.QPFX + q])[0]),
    # stronger ordinary baseline: the same divisive normalization + sum Psymem uses, on exact BM25 + exact cosine
    'hybrid_cc': lambda c, q: core._dn(c['bm25'].scores(q)) + core._dn(
        (d := c['E'] @ c['emb']([Embedder.QPFX + q])[0]) - np.median(d)),
    # ordinary search + neighbour expansion (the RAG "sentence window" trick), to price Psymem's S<->S links fairly
    'hybrid_cc_nb': lambda c, q: nb_spread(c, SCORERS['hybrid_cc'](c, q)),
}


def nb_spread(c, a, nb=0.5):
    sid = np.array([i[3] for i in c['items']], object); same = sid[1:] == sid[:-1]
    return a + nb * np.maximum(np.r_[0, a[:-1] * same], np.r_[a[1:] * same, 0])
PSY_INT = {'n_fly', 'fly_k', 'fan_in', 'seed', 'prf', 'prf_c', 'qpfx', 'K', 'signed', 'graded', 'qrate', 'mu_global'}
PSY_BUILD = {'n_fly', 'fly_k', 'fan_in', 'k1', 'b', 'seed', 'signed', 'graded', 'proj', 'mu_global'}


def psy_arg(k, v):  # proj=<name> -> models/psymem/<name>.npz; enc=<name> -> the spiking front end models/psymem/enc_<name>.pt
    if k == 'proj':
        return f'/home/glenn/projects/neuro/models/psymem/{v}.npz'
    return v if k == 'enc' else (int if k in PSY_INT else float)(v)


def psy_scorer(spec):
    """'psy' or 'psy:g_sem=0.8,prf=3' -> scorer; build keys go to Psymem(), qpfx picks the query prefix, enc swaps the bge
    front end for the spiking encoder (no query prefix), the rest go to activate()."""
    kw = {k: psy_arg(k, v) for k, v in (p.split('=') for p in spec.partition(':')[2].split(',') if p)}
    qpfx, enc = kw.pop('qpfx', 1), kw.pop('enc', None)
    bk = {k: kw.pop(k) for k in list(kw) if k in PSY_BUILD}
    key = ('psy', enc) + tuple(sorted(bk.items()))

    def f(c, q):
        if enc:
            import encoder
        emb = encoder.load(enc) if enc else c['emb']
        if key not in c:
            c[key] = core.Psymem(**bk)
            E = emb([i[1] for i in c['items']]) if enc else c['E']
            for (_, text, date, sid), e in zip(c['items'], E):
                c[key].remember(text, e, date, sid, who=text.split(':')[0])
        return c[key].activate(q, emb([(Embedder.QPFX if qpfx and not enc else '') + q])[0], now=c.get('now'), **kw)
    return f


def evaluate(mems, systems, emb, make_ctx=ctx_base):
    SCORERS.update({s: psy_scorer(s) for s in systems if s not in SCORERS})
    res = {s: collections.defaultdict(lambda: collections.defaultdict(list)) for s in systems}
    for m in mems:
        c = make_ctx(m, emb)
        ids, sess = [i[0] for i in m['items']], [i[3] for i in m['items']]
        for q, gold, cat, _, c['now'] in m['qs']:
            for s in systems:
                order = np.argsort(-SCORERS[s](c, q))[:max(KS)]
                for k in KS:
                    top = {ids[i] for i in order[:k]}
                    for key in (cat, 'ALL'):
                        if gold:
                            res[s][key][f'r@{k}'].append(len(top & set(gold)) / len(gold))
                            res[s][key][f'all@{k}'].append(float(set(gold) <= top))
                        if 'ans' in m:
                            res[s][key][f'sr@{k}'].append(len({sess[i] for i in order[:k]} & m['ans']) / len(m['ans']))
    return {s: {cat: {mt: (float(np.mean(v)), len(v)) for mt, v in d.items()} for cat, d in r.items()} for s, r in res.items()}


def table(out, metric):  # one row per system, one column per category
    first = out[next(iter(out))]
    cats = ['ALL'] + sorted((c for c in first if c != 'ALL'), key=str)
    lines = [f'-- {metric}'.ljust(44) + ''.join(f'{str(c)[:9]:>10}' for c in cats),
             'n'.rjust(44) + ''.join(f'{first[c][metric][1]:>10}' for c in cats)]
    for s in out:
        lines.append(f'{s[:44]:<44}' + ''.join(f"{100 * out[s][c].get(metric, (float('nan'),))[0]:>10.1f}" for c in cats))
    return '\n'.join(lines)


if __name__ == '__main__':
    which, systems = sys.argv[1], sys.argv[2:] or ['bm25', 'dense', 'hybrid']
    t0 = time.time()
    mems = load_locomo() if which == 'locomo' else load_lme()
    emb = Embedder(cache_file=D + f'emb_{which}.pkl')
    out = evaluate(mems, systems, emb)
    emb.save()
    metrics = ['r@5', 'r@10', 'all@10'] if which == 'locomo' else ['sr@5', 'r@10', 'all@10']
    print(f'== {which}: {len(mems)} memories, {len(systems)} systems, {time.time() - t0:.0f} s')
    print('\n'.join(table(out, mt) for mt in metrics), flush=True)
    tag = re.sub(r'[^\w.=-]+', '-', '_'.join(systems))[:120]
    json.dump(out, open(f'/home/glenn/projects/neuro/logs/n84_membench_{which}_{tag}.json', 'w'), indent=1)
