# psymem/abstain.py -- 2026-10-27 (Neuro 85): "knows what it doesn't know" (plan step 4, feature F4). Each system gives one
# confidence per question; AUROC = how well it separates answerable questions from unanswerable ones
# (LongMemEval-S *_abs questions; LoCoMo category 5 = adversarial, asked about the wrong person).
#   bm25 = top raw score;  dense = top cosine;  psy = Psymem.confidence (coverage of the cue's assemblies by one span).
# Usage: python psymem/abstain.py locomo|lme
import sys, time
import numpy as np
import bench, core


def auroc(pos, neg):  # P(confidence of an answerable question > an unanswerable one), ties count half
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + .5 * (pos == neg).mean())


if __name__ == '__main__':
    which, t0 = sys.argv[1], time.time()
    mems = bench.load_locomo() if which == 'locomo' else bench.load_lme(abs_too=True)
    emb = bench.Embedder(cache_file=bench.D + f'emb_{which}.pkl')
    conf = {s: ([], []) for s in ('bm25', 'dense', 'psy', 'psy_who')}  # (answerable, unanswerable)
    for m in mems:
        c = bench.ctx_base(m, emb)
        p = core.Psymem()
        for (_, text, date, sid), e in zip(c['items'], c['E']):
            p.remember(text, e, date, sid, who=text.split(':')[0])
        for q, gold, cat, *_ in m['qs']:
            qe = emb([bench.Embedder.QPFX + q])[0]
            neg = cat in (5, 'abstain')
            conf['bm25'][neg].append(c['bm25'].scores(q).max())
            conf['dense'][neg].append((c['E'] @ qe).max())
            conf['psy'][neg].append(p.confidence(q, qe, g_sem=0.7))
            conf['psy_who'][neg].append(p.confidence(q, qe, g_sem=0.7, g_who=0.6))
    emb.save()
    print(f'== abstain {which}: {len(conf["psy"][0])} answerable, {len(conf["psy"][1])} unanswerable, {time.time() - t0:.0f} s')
    for s, (pos, neg) in conf.items():
        print(f'{s:<8} AUROC {100 * auroc(pos, neg):5.1f}   mean conf answerable {np.mean(pos):.3f} / unanswerable {np.mean(neg):.3f}')
