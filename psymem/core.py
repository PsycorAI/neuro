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


class Psymem:
    def __init__(self, dim=384, n_fly=16384, fly_k=256, fan_in=24, k1=1.2, b=0.75, seed=0, signed=0, graded=0, proj=None,
                 mu_global=0):
        self.P, self.mu0 = learned_proj(proj) if proj else (flyhash_proj(n_fly, dim, fan_in, seed, signed), None)
        self.mu_global = mu_global  # centre on the encoder's training mean instead of this memory's own mean
        self.fly_k, self.k1, self.b, self.graded = fly_k, k1, b, graded  # graded: semantic synapse ~ the winner's rate
        self.diary, self.E, self.lex, self.hebb = [], [], {}, []  # hebb[s] = {lexical cue id: co-activation count}
        self.use_l, self.use_s = {}, {}  # learned-with-use synapses: (cue neuron, span) -> strength (lexical, semantic)
        self.dirty = True

    def remember(self, text, emb, date=None, sid=None, who=None):
        """Write one turn: diary append + Hebbian co-activation counts. Returns the span id. who = the speaker."""
        self.diary.append((text, date, sid, who)); self.E.append(np.asarray(emb, np.float32))
        self.hebb.append({self.lex.setdefault(w, len(self.lex)): n for w, n in bag(text).items()})
        self.dirty = True
        return len(self.diary) - 1

    def _compile(self):  # consolidate: apply the normalization rules; re-derive FlyHash codes (the mean adapts as memory grows)
        n = len(self.diary)
        rows = np.fromiter((c for h in self.hebb for c in h), np.int64)
        cols = np.repeat(np.arange(n), [len(h) for h in self.hebb])
        tf = np.fromiter((v for h in self.hebb for v in h.values()), np.float32)
        ln, deg = np.bincount(cols, tf, n), np.bincount(rows, minlength=len(self.lex))
        w = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln[cols] / ln.mean()))
        self.idf = np.log(1 + (n - deg + .5) / (deg + .5))
        w *= self.idf[rows]
        self.Wl = sp.csr_matrix((w, (rows, cols)), shape=(len(self.lex), n))
        use = lambda d, m: sp.csr_matrix((list(d.values()), tuple(np.array(list(d)).T)), shape=(m, n)) if d else sp.csr_matrix((m, n))
        self.Ul, self.Us = use(self.use_l, len(self.lex)), use(self.use_s, len(self.P))
        E = np.stack(self.E); self.mu = self.mu0 if self.mu_global else E.mean(0)
        codes = self._code(E)
        ws = np.ones(codes.shape, np.float32)
        if self.graded:  # Hebbian rate: pre (winner's drive above the k-WTA threshold) x post (1); mean 1 per span
            h = np.take_along_axis((E - self.mu) @ self.P.T, codes, 1)
            h -= h.min(1, keepdims=True) - 1e-6
            ws = h / h.mean(1, keepdims=True)
        self.Ws = sp.csr_matrix((ws.ravel(), (codes.ravel(), np.repeat(np.arange(n), self.fly_k))), shape=(len(self.P), n))
        self.sid = np.array([d[2] for d in self.diary], object)
        self.days = np.array([day(d[1]) if d[1] else 0 for d in self.diary])  # the time cell each span is wired to
        # who cells: one per speaker; a span is wired to its speaker's cell and to the cells of speakers it names
        self.whos = sorted({stem(d[3].lower()) for d in self.diary if d[3]})
        self.who_w = np.array([[d[3] is not None and (stem(d[3].lower()) == w or w in bag(d[0])) for d in self.diary]
                               for w in self.whos], bool).reshape(len(self.whos), n)
        self.dirty = False

    def _who(self, cue, g_who):  # who-cell gate: spans bound to a person the cue names stay at 1, others drop to 1 - g_who
        named = [i for i, w in enumerate(self.whos) if w in bag(cue)]
        return 1 - g_who + g_who * self.who_w[named].any(0) if named and g_who else 1.

    def _code(self, E):  # FlyHash: expand, k-winners-take-all -> active neuron ids (n x fly_k)
        return np.argpartition((E - self.mu) @ self.P.T, -self.fly_k, axis=1)[:, -self.fly_k:]

    def _sweep(self, lex_ids, code, g_sem):
        al = np.asarray(self.Wl[lex_ids].sum(0)).ravel() if len(lex_ids) else np.zeros(self.Wl.shape[1])
        a_s = np.asarray(self.Ws[code].sum(0)).ravel()
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
        deg = np.array([self.Wl[self.lex[w]].nnz if w in self.lex else 0 for w in words])
        cap = np.log(1 + (n - deg + .5) / (deg + .5)).sum()
        cov = np.asarray(self.Wl[[self.lex[w] for w in words if w in self.lex]].sum(0)).ravel() / cap
        ov = np.asarray(self.Ws[self._code(emb[None])[0]].sum(0)).ravel() / self.fly_k
        return float(((np.minimum(cov, 1) + g_sem * ov) / (1 + g_sem) * self._who(cue, g_who)).max())

    def reinforce(self, cue, emb, spans, lr=1.):
        """Learn with use: a confirmed recall wires this cue's active neurons (words, FlyHash winners) to the confirmed spans.
        New words get new cue neurons. (Codes use the current mean; dreaming would re-derive them as the memory grows.)"""
        if self.dirty:
            self._compile()
        ids, code = [self.lex.setdefault(w, len(self.lex)) for w in bag(cue)], self._code(emb[None])[0]
        for s in spans:
            for k, d in [((c, s), self.use_l) for c in ids] + [((j, s), self.use_s) for j in code]:
                d[k] = d.get(k, 0.) + lr
        self.dirty = True

    def activate(self, cue, emb, now=None, g_sem=1.0, prf=0, prf_c=10, beta=0.3, nb=0.0, g_t=0.0, g_who=0.0, adapt=0.0, K=20,
                 g_use=0.0, qrate=0):
        """Spread one cue through the synapses; returns one activation per diary turn. now = date string of 'today'.
        qrate: the cue's FlyHash winners fire at a rate (spike count) set by their drive above the k-WTA threshold."""
        if self.dirty:
            self._compile()
        lex_ids, code = [self.lex[w] for w in bag(cue) if w in self.lex], self._code(emb[None])[0]
        Ml, Ms = self.Wl[lex_ids].toarray(), self.Ws[code].toarray()  # this cue's synaptic rows
        gl, gs = np.ones(len(Ml)), np.ones(len(Ms))  # cue-neuron gains (adaptation lowers them)
        if qrate:
            h = (emb - self.mu) @ self.P[code].T
            gs = (h - h.min() + 1e-6) / (h - h.min() + 1e-6).mean()
        a_s0 = gs @ Ms
        zl, med = (gl @ Ml).max() or 1., np.median(a_s0)
        zs = (a_s0.max() - med) or 1.

        def drive():  # one C->S sweep, divisively normalized by the first sweep's peaks
            return np.maximum(gl @ Ml, 0) / zl + g_sem * np.maximum(gs @ Ms - med, 0) / zs
        extra, gate = 0., 1.
        a = drive()
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

        def post(a):  # PRF, time gating, then S<->S links to the previous / next turn of the same session
            a = (a + extra) * gate
            return a + nb * np.maximum(np.r_[0, a[:-1] * same], np.r_[a[1:] * same, 0]) if nb else a
        a = post(a)
        if adapt:
            out, picked = a.copy(), []
            for r in range(K):
                s = int(np.argmax(np.where(np.isin(np.arange(len(a)), picked), -np.inf, a)))
                picked.append(s); out[s] = 1e3 + K - r
                gl[Ml[:, s] > 0] *= 1 - adapt; gs[Ms[:, s] > 0] *= 1 - adapt
                a = post(drive())
            rest = np.setdiff1d(np.arange(len(a)), picked)
            out[rest] = a[rest]
            a = out
        return a
