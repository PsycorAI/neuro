# psymem/core.py -- 2026-10-27 (Neuro 85): Psymem v0, a two-layer synaptic associative memory over a diary.
# Diary = exact turns with dates (the master copy). Cue neurons C -> span neurons S (one per diary turn):
#   lexical cues  = one neuron per word stem. Synapse = saturating Hebbian co-activation count, scaled by heterosynaptic
#                   normalization (a busy cue neuron shares its resource over many targets -> IDF) and by the span's shared
#                   postsynaptic resource (-> length normalization). With k1 / b as in BM25 this channel IS BM25 (disclosed).
#   semantic cues = FlyHash (Dasgupta et al. 2017): sparse binary random expansion of the mean-centred embedding, then
#                   k-winners-take-all; a 1-bit Willshaw synapse from each winner to the span.
# Recall = cue spikes sweep C->S once (accumulate-only); each channel is divisively normalized; summed with gain g_sem.
# Step 3 mechanisms (all off by default):
#   prf   = S->C->S pattern completion from the k-WTA top spans;   nb = S<->S links to temporal neighbours in the same session;
#   g_t   = time cells: each span is wired to the time cell of the day it was stored; a time expression in the cue ("a week
#           ago", "in March", "last weekend") drives time cells around that day, which gate span activity (1 + g_t * t);
#   adapt = cue adaptation (explaining away): recall K spans one at a time; cue neurons a recalled span already explains
#           fatigue by this fraction, so the next recall favours the cues not yet covered (multi-evidence questions).
# Writes need no LLM calls. CPU numpy / scipy.sparse.
# Storage (10-28): synapses live in column blocks (one column per span). Spans not yet frozen sit in a small live tail
# (meaning codes kept dense); every B_FREEZE new spans freeze into a block, and equal-size neighbour blocks merge up to
# B_CAP spans. Word synapses keep raw co-activation counts; the BM25 normalization (IDF, length) is applied at recall to
# the cue's rows only. A write therefore costs about the same at any memory size (before: Wl / Ws rebuilt whole every
# write, 2.5 s at 10k spans). Recall results are identical.
import re, functools, collections, datetime
import numpy as np, scipy.sparse as sp
from dateutil import parser as dparser
from nltk.stem import PorterStemmer

STOP = set('a an the and or but if of to in on at for with is are was were be been am i you he she it we they me my your '
           'his her its our their this that these those do does did have has had not no so as by from what when where who '
           'which how why will would can could should about'.split())
_ps = PorterStemmer()


@functools.lru_cache(maxsize=None)
def stem(w):
    return _ps.stem(w)


@functools.lru_cache(maxsize=300000)
def bag(text):
    return collections.Counter(stem(w) for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOP)


def _dn(x):  # divisive normalization: rectify, scale the peak to 1
    x = np.maximum(x, 0); m = x.max()
    return x / m if m > 0 else x


@functools.lru_cache(maxsize=None)
def day(s):  # date string -> day number
    return dparser.parse(re.sub(r'\(\w+\)', '', s), fuzzy=True).toordinal()


MONTHS = 'january february march april may june july august september october november december'.split()
NUM = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12,
           couple=2, few=3, several=3)
UNIT = dict(day=1, week=7, month=30, year=365)
_N = r'(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|several)'
_M = '(' + '|'.join(MONTHS) + ')'


def _month(mo, yr, now):  # (first, last) day of a month; with no year, the latest such month not after now
    d = datetime.date.fromordinal(now)
    y = int(yr) if yr else d.year - (mo > d.month)
    nxt = datetime.date(y + mo // 12, mo % 12 + 1, 1)
    return datetime.date(y, mo, 1).toordinal(), nxt.toordinal() - 1


def when(cue, now):
    """Time expression in a cue -> (first day, last day) relative to day `now`, or None."""
    t, wd = cue.lower().replace('a couple of', 'couple').replace('a few', 'few'), datetime.date.fromordinal(now).weekday()
    num = lambda s: int(s) if s.isdigit() else NUM[s]
    if m := re.search(_N + r' (day|week|month|year)s? ago', t):
        k, u = num(m[1]) * UNIT[m[2]], UNIT[m[2]]
        return now - k - max(1, u // 2, k // 7), now - k + max(1, u // 2, k // 7)
    if 'yesterday' in t:
        return now - 1, now - 1
    if m := re.search(r'(?:past|last|previous) ' + _N + r' (day|week|month|year)s?', t):
        return now - num(m[1]) * UNIT[m[2]], now
    if re.search(r'(?:past|last|previous) weekend', t):
        sat = now - ((wd - 5) % 7 or 7)
        return sat, sat + 1
    if re.search(r'(?:past|last|previous) week\b', t):
        return now - wd - 7, now - wd - 1
    if re.search(r'(?:past|last|previous) month\b', t):
        d = datetime.date.fromordinal(now)
        return _month(12 if d.month == 1 else d.month - 1, d.year - (d.month == 1), now)
    if m := re.search(r'(first|second|third|fourth|last) week of ' + _M + r'(?:,? (\d{4}))?', t):
        lo, hi = _month(MONTHS.index(m[2]) + 1, m[3], now)
        i = 'first second third fourth'.split().index(m[1]) if m[1] != 'last' else None
        return (hi - 6, hi) if i is None else (lo + 7 * i, lo + 7 * i + 6)
    if (m := re.search(_M + r' (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})', t)) or (m := re.search(r'(\d{1,2}) ' + _M + r',? (\d{4})', t)):
        g = m.groups() if m[1] in MONTHS else (m[2], m[1], m[3])
        d = datetime.date(int(g[2]), MONTHS.index(g[0]) + 1, int(g[1])).toordinal()
        return d, d
    if m := re.search(r'\b(?:in|during|of|since) ' + _M + r'(?:,? (\d{4}))?', t):
        return _month(MONTHS.index(m[1]) + 1, m[2], now)
    if m := re.search(r'(?:last|on|this past) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)', t):
        back = (wd - 'monday tuesday wednesday thursday friday saturday sunday'.split().index(m[1])) % 7 or 7
        return now - back, now - back
    return None


@functools.lru_cache(maxsize=16)
def flyhash_proj(n_fly, dim, fan_in, seed, signed):  # each Kenyon-cell-like neuron sums fan_in random embedding dims
    rng, P = np.random.default_rng(seed), np.zeros((n_fly, dim), np.float32)
    P[np.arange(n_fly)[:, None], rng.random((n_fly, dim)).argsort(1)[:, :fan_in]] = 1
    return P * rng.choice([-1, 1], P.shape) if signed else P  # signed = excitatory + inhibitory inputs


@functools.lru_cache(maxsize=4)
def learned_proj(path):  # a learned expansion layer (psymem/hebb.py): P (n_fly x dim) and the training-corpus mean
    d = np.load(path)
    return d['P'], d['mu']


B_FREEZE, B_CAP = 256, 16384  # live tail freezes every 256 spans; equal-size neighbour blocks merge up to 16,384 spans


def _hcat(ms, rows):  # side-by-side concat of CSR matrices with equal rows, O(nnz) (scipy's hstack goes through COO)
    ms = [m for m in ms if m.shape[1]]
    if len(ms) < 2:
        return ms[0] if ms else sp.csr_matrix((rows, 0))
    ip = np.sum([m.indptr.astype(np.int64) for m in ms], 0)
    data, idx = np.empty(ip[-1], ms[0].dtype), np.empty(ip[-1], np.int32)
    cur, c0 = ip[:-1].copy(), 0
    for m in ms:
        cnt = np.diff(m.indptr)
        pos = np.arange(m.nnz) + np.repeat(cur - m.indptr[:-1], cnt)
        data[pos], idx[pos] = m.data, m.indices + c0
        cur += cnt; c0 += m.shape[1]
    return sp.csr_matrix((data, idx, ip), shape=(rows, c0))


class Psymem:
    def __init__(self, dim=384, n_fly=16384, fly_k=256, fan_in=24, k1=1.2, b=0.75, seed=0, signed=0, graded=0, proj=None,
                 mu_global=0):
        self.P, self.mu0 = learned_proj(proj) if proj else (flyhash_proj(n_fly, dim, fan_in, seed, signed), None)
        self.mu_global = mu_global  # centre on the encoder's training mean instead of this memory's own mean
        self.fly_k, self.k1, self.b, self.graded = fly_k, k1, b, graded  # graded: semantic synapse ~ the winner's rate
        self.diary, self.E, self.lex, self.hebb = [], [], {}, []  # hebb[s] = {lexical cue id: co-activation count}
        self.use_l, self.use_s = {}, {}  # learned-with-use synapses: (cue neuron, span) -> strength (lexical, semantic)
        self._deg = np.zeros(0, np.int64)  # per word neuron: how many spans it is wired to (-> IDF)
        self._esum, self._nl = np.zeros(dim), 0  # running embedding sum (drift test); word-synapse count
        self._wsp = {}  # who cell -> spans wired to it (the speaker's own turns + turns that name them)
        self._lb, self._sb, self._nf, self._nc = [], [], 0, 0  # frozen word / meaning blocks; spans frozen; spans coded
        self._mu_c = None  # the centring mean all current meaning codes were made with
        self._tc, self._tw = np.zeros((0, fly_k), np.int32), np.zeros((0, fly_k), np.float32)  # live tail codes, weights
        self.sid, self.days, self.tlen, self.isq, self.ln = (np.zeros(0, object), np.zeros(0, int), np.zeros(0, int),
                                                             np.zeros(0, bool), np.zeros(0))
        self.dirty = True

    def remember(self, text, emb, date=None, sid=None, who=None):
        """Write one turn: diary append + Hebbian co-activation counts. Returns the span id. who = the speaker."""
        self.diary.append((text, date, sid, who)); self.E.append(np.asarray(emb, np.float32))
        h = {self.lex.setdefault(w, len(self.lex)): n for w, n in bag(text).items()}
        self.hebb.append(h)
        if len(self.lex) > len(self._deg):  # grow by doubling (amortized O(1))
            self._deg = np.r_[self._deg, np.zeros(max(len(self.lex), 2 * len(self._deg)) - len(self._deg), np.int64)]
        self._deg[list(h)] += 1
        self._esum += self.E[-1]; self._nl += len(h)
        s = len(self.diary) - 1
        if who:  # who cells: a new speaker also binds the earlier turns that name them (once per new speaker)
            w = stem(who.lower())
            if w not in self._wsp:
                self._wsp[w] = [i for i, d in enumerate(self.diary[:-1]) if d[3] is not None and w in bag(d[0])]
            for v, ids in self._wsp.items():
                if v == w or v in bag(text):
                    ids.append(s)
        self.dirty = True
        return s

    def _codes_of(self, E):  # FlyHash winners + graded weights for spans E, with the current centring mean self.mu
        codes = np.empty((len(E), self.fly_k), np.int32); ws = np.ones((len(E), self.fly_k), np.float32)
        # in blocks of 1,024 spans: the full n x n_fly drive matrix was 7.9 GB at 30k spans (OOM, 10-27 night)
        for i in range(0, len(E), 1024):
            X = (E[i:i + 1024] - self.mu) @ self.P.T
            c = np.argpartition(X, -self.fly_k, axis=1)[:, -self.fly_k:]
            codes[i:i + 1024] = c
            if self.graded:  # Hebbian rate: pre (winner's drive above the k-WTA threshold) x post (1); mean 1 per span
                h = np.take_along_axis(X, c, 1)
                h -= h.min(1, keepdims=True) - 1e-6
                ws[i:i + 1024] = h / h.mean(1, keepdims=True)
            del X
        return codes, ws

    def _word_block(self, a, z):  # raw co-activation counts of spans a..z-1 (word neurons x spans)
        hs = self.hebb[a:z]
        rows = np.fromiter((c for h in hs for c in h), np.int64)
        tf = np.fromiter((v for h in hs for v in h.values()), np.float32)
        return sp.csr_matrix((tf, (rows, np.repeat(np.arange(len(hs)), [len(h) for h in hs]))), shape=(len(self.lex), len(hs)))

    def _freeze(self, m):  # the first m live-tail spans become one frozen block; equal-size neighbours merge up to B_CAP
        c, w = self._tc[:m], self._tw[:m]
        self._sb.append(sp.csr_matrix((w.ravel(), (c.ravel(), np.repeat(np.arange(m), self.fly_k))), shape=(len(self.P), m)))
        self._lb.append(self._word_block(self._nf, self._nf + m))
        self._tc, self._tw, self._nf = self._tc[m:], self._tw[m:], self._nf + m
        while len(self._sb) > 1 and self._sb[-1].shape[1] == self._sb[-2].shape[1] < B_CAP:
            self._sb[-2:] = [_hcat(self._sb[-2:], len(self.P))]
            for b in self._lb[-2:]:
                b.resize((len(self.lex), b.shape[1]))
            self._lb[-2:] = [_hcat(self._lb[-2:], len(self.lex))]

    def _compile(self):  # consolidate: apply the normalization rules; code new spans (all spans if the mean drifted)
        n, V = len(self.diary), len(self.lex)
        new = self.diary[len(self.days):]  # per-span facts, appended for new spans only
        self.sid = np.r_[self.sid, np.array([d[2] for d in new] + [None], object)[:-1]]
        self.days = np.r_[self.days, np.array([day(d[1]) if d[1] else 0 for d in new], int)]  # the time cell each span is wired to
        self.tlen = np.r_[self.tlen, np.array([len(d[0]) for d in new], int)]  # turn length (chars) and is-a-question, for gated S<->S links
        self.isq = np.r_[self.isq, np.array([d[0].rstrip(' "\'').endswith('?') for d in new], bool)]
        self.ln = np.r_[self.ln, np.array([sum(h.values()) for h in self.hebb[len(self.ln):]], float)]  # BM25 span length
        self._lnm = self.ln.mean()
        if len(self._deg) < V:  # words grown by reinforce()
            self._deg = np.r_[self._deg, np.zeros(V - len(self._deg), np.int64)]
        self.deg = self._deg[:V]
        self.idf = np.log(1 + (n - self.deg + .5) / (self.deg + .5))
        use = lambda d, m: sp.csr_matrix((list(d.values()), tuple(np.array(list(d)).T)), shape=(m, n)) if d else sp.csr_matrix((m, n))
        self.Ul, self.Us = use(self.use_l, V), use(self.use_s, len(self.P))
        # incremental (10-28): while the centring mean drifts < 5%, keep earlier spans' codes and code only the new spans
        # (with the same mean, so cue codes stay consistent); a larger drift re-derives all codes ("dreaming")
        # All codes always share one mean, so (diary, _mu_c) fixes the state: a memory file restores _mu_c and reloads exact.
        mu = self.mu0 if self.mu_global else self._esum / n
        if self._mu_c is None or np.linalg.norm(mu - self._mu_c) > 0.05 * (np.linalg.norm(self._mu_c) or 1.):
            self._mu_c, self._nc = self.mu0 if self.mu_global else np.stack(self.E).mean(0), 0
        start = self._nc
        if not start:
            self._lb, self._sb, self._nf = [], [], 0
            self._tc, self._tw = self._tc[:0], self._tw[:0]
        self.mu = self._mu_c
        if start < n:
            c, w = self._codes_of(np.stack(self.E[start:]))
            self._tc, self._tw = np.r_[self._tc, c], np.r_[self._tw, w]
        self._nc = n
        if len(self._tc) >= B_FREEZE:
            self._freeze(len(self._tc) // B_FREEZE * B_FREEZE)
        self._lt = self._word_block(self._nf, n)
        self.nnz = self._nl + n * self.fly_k  # synapses (word + meaning)
        self._full = {}
        # who cells: one per speaker; a span is wired to its speaker's cell and to the cells of speakers it names
        self.whos = sorted(self._wsp)
        self.who_w = np.zeros((len(self.whos), n), bool)
        for i, w in enumerate(self.whos):
            self.who_w[i, self._wsp[w]] = True
        self.dirty = False

    def word_rows(self, ids):  # BM25-weighted synapses of word neurons ids (len(ids) x spans), normalized at recall
        ids, V = np.asarray(ids, np.int64), len(self.lex)
        for b in self._lb:
            if b.shape[0] < V:  # word neurons grown since this block froze
                b.resize((V, b.shape[1]))
        M = _hcat([b[ids] for b in self._lb + [self._lt]], len(ids))
        tf, cols = M.data, M.indices
        w = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.ln[cols] / self._lnm))
        w *= self.idf[ids[np.repeat(np.arange(len(ids)), np.diff(M.indptr))]]
        return sp.csr_matrix((w, cols, M.indptr), shape=M.shape)

    def _meaning_parts(self, code):  # synapses of meaning neurons code, one CSR per frozen block + the live tail, in span order
        parts = [b[code] for b in self._sb]
        if len(self._tc):
            pos = np.full(len(self.P), -1); pos[code] = np.arange(len(code))
            r = pos[self._tc]; s, j = np.nonzero(r >= 0)
            parts.append(sp.csr_matrix((self._tw[s, j], (r[s, j], s)), shape=(len(code), len(self._tc))))
        return parts

    def meaning_rows(self, code):  # the same as one matrix (len(code) x spans); recall itself skips the concatenation
        return _hcat(self._meaning_parts(code), len(code))

    @staticmethod
    def _mv(parts, x):  # (rows of all parts)^T @ x, block by block -> one value per span
        return np.concatenate([m.T @ x for m in parts]) if parts else np.zeros(0)

    @staticmethod
    def _colsum(parts):  # column sums, block by block (same arithmetic as summing the concatenated matrix)
        return np.concatenate([np.asarray(m.sum(0)).ravel() for m in parts]) if parts else np.zeros(0)

    @property
    def Wl(self):  # the whole word-synapse matrix (only pattern completion / analysis need it; built on demand)
        if 'l' not in self._full:
            self._full['l'] = self.word_rows(np.arange(len(self.lex)))
        return self._full['l']

    @property
    def Ws(self):  # the whole meaning-synapse matrix, on demand
        if 's' not in self._full:
            self._full['s'] = self.meaning_rows(np.arange(len(self.P)))
        return self._full['s']

    def _who(self, cue, g_who):  # who-cell gate: spans bound to a person the cue names stay at 1, others drop to 1 - g_who
        named = [i for i, w in enumerate(self.whos) if w in bag(cue)]
        return 1 - g_who + g_who * self.who_w[named].any(0) if named and g_who else 1.

    def _code(self, E):  # FlyHash: expand, k-winners-take-all -> active neuron ids (n x fly_k)
        return np.argpartition((E - self.mu) @ self.P.T, -self.fly_k, axis=1)[:, -self.fly_k:]

    def _sweep(self, lex_ids, code, g_sem):
        al = np.asarray(self.word_rows(lex_ids).sum(0)).ravel() if len(lex_ids) else np.zeros(len(self.diary))
        a_s = self._colsum(self._meaning_parts(code))
        return _dn(al) + g_sem * _dn(a_s - np.median(a_s))

    def _time(self, lo, hi, now):  # time-cell drive: 1 inside [lo, hi], Gaussian fall-off, wider further back (scalar timing)
        dist = np.maximum(np.maximum(lo - self.days, self.days - hi), 0)
        sig = max(1., (hi - lo + 1) / 4) + 0.05 * max(now - hi, 0)
        return np.exp(-dist ** 2 / (2 * sig ** 2))

    def confidence(self, cue, emb, g_sem=1.0, g_who=0.0):
        """How sure, 0..1: the best span's share of this cue's possible drive (coincidence of its cue assemblies).
        Lexical: synaptic drive received / the cue's total IDF (words never stored count at full IDF, so they lower it);
        semantic: share of the cue's FlyHash winners wired to the span."""
        if self.dirty:
            self._compile()
        n, words = len(self.diary), list(bag(cue))
        if not words:
            return 0.
        deg = np.array([self.deg[self.lex[w]] if w in self.lex else 0 for w in words])
        cap = np.log(1 + (n - deg + .5) / (deg + .5)).sum()
        cov = np.asarray(self.word_rows([self.lex[w] for w in words if w in self.lex]).sum(0)).ravel() / cap
        ov = self._colsum(self._meaning_parts(self._code(emb[None])[0])) / self.fly_k
        return float(((np.minimum(cov, 1) + g_sem * ov) / (1 + g_sem) * self._who(cue, g_who)).max())

    def reinforce(self, cue, emb, spans, lr=1.):
        """Learn with use: a confirmed recall wires this cue's active neurons (words, FlyHash winners) to the confirmed spans.
        New words get new cue neurons. (Codes use the current mean; dreaming would re-derive them as the memory grows.)"""
        if self.dirty:
            self._compile()
        return self.wire(list(bag(cue)), [int(j) for j in self._code(emb[None])[0]], spans, lr)

    def wire(self, words, code, spans, lr=1.):  # the synaptic part of reinforce(); also replays it from a memory file
        ids = [self.lex.setdefault(w, len(self.lex)) for w in words]
        for s in spans:
            for k, d in [((c, s), self.use_l) for c in ids] + [((j, s), self.use_s) for j in code]:
                d[k] = d.get(k, 0.) + lr
        self.dirty = True
        return words, code

    def activate(self, cue, emb, now=None, g_sem=1.0, prf=0, prf_c=10, beta=0.3, nb=0.0, g_t=0.0, g_who=0.0, adapt=0.0, K=20,
                 g_use=0.0, qrate=0, nb_mode=0, nb_len=300):
        """Spread one cue through the synapses; returns one activation per diary turn. now = date string of 'today'.
        qrate: the cue's FlyHash winners fire at a rate (spike count) set by their drive above the k-WTA threshold.
        nb_mode (10-27 night, gated S<->S links): 0 = both neighbours, every turn; 1 = only turns shorter than nb_len chars
        receive neighbour support (long turns stand alone); 2 = question -> answer links only (a turn takes its previous
        turn's activity if that turn asked a question, and a question takes its answer's); 3 = support scaled by
        min(1, nb_len / length)."""
        if self.dirty:
            self._compile()
        lex_ids, code = [self.lex[w] for w in bag(cue) if w in self.lex], self._code(emb[None])[0]
        # this cue's synaptic rows, kept sparse (10-28: dense rows made recall O(fly_k x n): 378 ms at 20k spans)
        Ml, Sp = self.word_rows(lex_ids), self._meaning_parts(code)  # meaning rows stay per block (no concatenation)
        MlT = Ml.T  # CSC view: fast matvec, no conversion (tocsr was 70% of recall time)
        gl, gs = np.ones(MlT.shape[1]), np.ones(len(code))  # cue-neuron gains (adaptation lowers them)
        if qrate:
            h = (emb - self.mu) @ self.P[code].T
            gs = (h - h.min() + 1e-6) / (h - h.min() + 1e-6).mean()
        a_l0, a_s0 = MlT @ gl, self._mv(Sp, gs)
        zl, med = a_l0.max(initial=0) or 1., np.median(a_s0)
        zs = (a_s0.max() - med) or 1.

        def drive(al=None, as_=None):  # one C->S sweep, divisively normalized by the first sweep's peaks
            al = MlT @ gl if al is None else al
            as_ = self._mv(Sp, gs) if as_ is None else as_
            return np.maximum(al, 0) / zl + g_sem * np.maximum(as_ - med, 0) / zs
        extra, gate = 0., 1.
        a = drive(a_l0, a_s0)  # the first sweep is the one already computed
        if prf:  # pattern completion: the k-WTA top spans drive their cue neurons back (S->C), which sweep again (C->S)
            top = np.argpartition(-a, prf)[:prf]
            dl, ds = self.Wl[:, top] @ a[top], self.Ws[:, top] @ a[top]
            cl = np.argsort(-dl)[:prf_c]
            extra = beta * (1 + g_sem) * _dn(self._sweep(cl[dl[cl] > 0], np.argsort(-ds)[:self.fly_k], g_sem))
        if g_use and (self.use_l or self.use_s):  # learned-with-use synapses: IDF share of the cue's words + FlyHash overlap
            idf = self.idf[lex_ids]
            extra = extra + g_use * (idf @ self.Ul[lex_ids].toarray() / (idf.sum() or 1.)
                                     + g_sem * np.asarray(self.Us[code].sum(0)).ravel() / self.fly_k)
        if g_t:
            t_now = day(now) if now else int(self.days.max())
            if w := when(cue, t_now):
                gate = 1 + g_t * self._time(*w, t_now)
        gate = gate * self._who(cue, g_who)
        same = self.sid[1:] == self.sid[:-1]
        fp = np.r_[0, same].astype(float)  # link from the previous turn
        fn = np.r_[same, 0].astype(float)  # link from the next turn
        if nb_mode == 1:
            fp, fn = fp * (self.tlen < nb_len), fn * (self.tlen < nb_len)
        elif nb_mode == 2:
            fp, fn = fp * np.r_[False, self.isq[:-1]], fn * self.isq
        elif nb_mode == 3:
            sc = np.minimum(1, nb_len / np.maximum(self.tlen, 1))
            fp, fn = fp * sc, fn * sc

        def post(a):  # PRF, time gating, then S<->S links to the previous / next turn of the same session
            a = (a + extra) * gate
            return a + nb * np.maximum(np.r_[0, a[:-1]] * fp, np.r_[a[1:], 0] * fn) if nb else a
        a = post(a)
        if adapt:
            Ms, out, picked = _hcat(Sp, len(code)), a.copy(), []
            for r in range(K):
                s = int(np.argmax(np.where(np.isin(np.arange(len(a)), picked), -np.inf, a)))
                picked.append(s); out[s] = 1e3 + K - r
                gl[Ml[:, [s]].nonzero()[0]] *= 1 - adapt; gs[Ms[:, [s]].nonzero()[0]] *= 1 - adapt  # cue neurons wired to s
                a = post(drive())
            rest = np.setdiff1d(np.arange(len(a)), picked)
            out[rest] = a[rest]
            a = out
        return a
