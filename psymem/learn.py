# psymem/learn.py -- 2026-10-27 (Neuro 85): "gets better with use" (memory product plan step 5, feature F5).
# Per memory the questions alternate into A (asked, then feedback) and B (never asked).
#   before:   r@10 on paraphrases of A and on A / B as written; Psymem and ordinary search (hybrid_cc_nb, which never learns)
#   feedback: each A question is asked once; the user confirms the right turns -> Psymem.reinforce(question, turns)
#             confirm = only the gold turns Psymem showed in its top 10;  correct = every gold turn (the user points to it)
#   after:    paraphrases of A (did it learn?), A again, B (did learning hurt anything else?)
# Paraphrases: data/membench/para_{which}.json (Llama-3.2-3B). Feedback is simulated from gold evidence labels (disclosed).
# LongMemEval has one question per memory, so B is empty there. Usage: python psymem/learn.py locomo|lme [psy config]
import sys, json, time, collections
import numpy as np
import bench, core

if __name__ == '__main__':
    which, t0 = sys.argv[1], time.time()
    kw = {k: (int if k in bench.PSY_INT else float)(v)
          for k, v in (p.split('=') for p in (sys.argv[2] if len(sys.argv) > 2 else 'g_sem=0.7,nb=0.5,g_t=1').split(','))}
    bk = {k: kw.pop(k) for k in list(kw) if k in bench.PSY_BUILD}
    kw.setdefault('g_use', 1.0)
    # keep only rewrites that are questions: the 3B model sometimes answered instead (leaking answer words into the cue)
    para = {q: [x for x in v if x.endswith('?')] for q, v in json.load(open(bench.D + f'para_{which}.json')).items()}
    mems = bench.load_locomo() if which == 'locomo' else bench.load_lme()
    emb = bench.Embedder(cache_file=bench.D + f'emb_{which}.pkl')
    QE = lambda q: emb([bench.Embedder.QPFX + q])[0]
    res = collections.defaultdict(list)
    for m in mems:
        c, ids = bench.ctx_base(m, emb), {i[0]: n for n, i in enumerate(m['items'])}
        A, B = m['qs'][0::2], m['qs'][1::2]
        nets = {}
        for mode in ('confirm', 'correct'):
            nets[mode] = p = core.Psymem(**bk)
            for (_, text, date, sid), e in zip(m['items'], c['E']):
                p.remember(text, e, date, sid, who=text.split(':')[0])

        def r10(scores, gold):
            top = set(np.argsort(-scores)[:10])
            return sum(ids[g] in top for g in gold) / len(gold)

        def score_all(tag, p):  # p = None -> ordinary search
            f = (lambda q, now: bench.SCORERS['hybrid_cc_nb'](c, q)) if p is None else \
                (lambda q, now: p.activate(q, QE(q), now=now, **kw))
            for q, gold, _, _, now in A:
                res[f'A paraphrases | {tag}'] += [r10(f(x, now), gold) for x in para.get(q, [])]
                res[f'A as asked    | {tag}'].append(r10(f(q, now), gold))
            for q, gold, _, _, now in B:
                res[f'B never asked | {tag}'].append(r10(f(q, now), gold))
        score_all('ordinary search', None)
        score_all('psymem before', nets['confirm'])
        for q, gold, _, _, now in A:
            top = set(np.argsort(-nets['confirm'].activate(q, QE(q), now=now, **kw))[:10])
            nets['confirm'].reinforce(q, QE(q), [ids[g] for g in gold if ids[g] in top])
            nets['correct'].reinforce(q, QE(q), [ids[g] for g in gold])
        for mode, p in nets.items():
            score_all(f'psymem after {mode}', p)
    emb.save()
    print(f'== learn-with-use {which}: {len(mems)} memories, config {bk} {kw}, {time.time() - t0:.0f} s')
    for k in sorted(res):
        print(f'{k:<45} r@10 {100 * np.mean(res[k]):5.1f}   n {len(res[k])}')
