# n85_memdemo_conf.py -- 2026-10-27 (Neuro 85): confidence on the demo diary. Benchmarks (logs/n85_abstain2.log): psy_who x
# unk (IDF share of cue words that have a word neuron) is the best Psymem signal on both (LoCoMo 60.4, LME 72.9 AUROC).
# Compare engine confidence variants on answerable vs unanswerable demo questions.
import sys, os
import numpy as np
sys.path.insert(0, '/home/glenn/projects/neuro/scripts/memdemo')
import engine, core
f = '/home/glenn/projects/neuro/logs/memdemo/probe.plasmem'
os.path.exists(f) and os.remove(f)
E = engine.Engine('probe'); p = E.p
A = ["Where is Lena moving?", "What did Sam fix?", "What happened at the dentist last week?", "What will decorate the hallway?",
     "What did they see from the top?", "Who helped Maya with her login?", "What is Maya learning on Wednesdays?",
     "What did Sam try to make?", "When is the dentist appointment?", "Who pays for Lena's studies?", "How much are the flights to Lisbon?",
     "When is Sam's mom visiting?", "What ice cream does Maya like?", "What did Maya climb at the gym?", "Who is Maya's pottery teacher?",
     "How long was the hike?", "What did Lena study?", "When does Lena leave for Portugal?", "What did Sam buy at the market?",
     "What went wrong with the soup?"]
U = ["What is Maya's cat called?", "What car does Sam drive?", "Where did Maya go to college?", "What is Lena's favourite movie?",
     "How old is Sam?", "What is Sam's job?", "Where does Sam's brother live?", "What is Maya's favourite band?",
     "Did Maya run a marathon?", "What language is Sam learning?", "Where did they go skiing?", "What is Lena's boyfriend's name?",
     "How much is their rent?", "What did Maya get for her birthday?", "Which hospital was Sam born in?", "What game did Sam play?"]


def unk(q):
    words, n = list(core.bag(q)), len(p.diary)
    if not words:
        return 0.
    idf = np.array([p.idf[p.lex[w]] if w in p.lex else np.log(1 + (n + .5) / .5) for w in words])
    return float(idf[[w in p.lex for w in words]].sum() / idf.sum())


rows = {k: ([], []) for k in ('conf', 'unk', 'conf_unk')}
for neg, qs in ((0, A), (1, U)):
    for q in qs:
        e = E.qemb(q)
        c = p.confidence(q, e, g_sem=1., g_who=engine.G_WHO); u = unk(q)
        for k, v in (('conf', c), ('unk', u), ('conf_unk', c * u)):
            rows[k][neg].append(v)
        print(f"{'U' if neg else 'A'} conf {c:.2f} unk {u:.2f} c*u {c * u:.2f} | {q}")
for k, (a, u) in rows.items():
    a, u = np.array(a)[:, None], np.array(u)[None, :]
    print(k, 'AUROC', round(100 * ((a > u).mean() + .5 * (a == u).mean()), 1), 'A range', a.min().round(2), a.max().round(2), 'U range', u.min().round(2), u.max().round(2))
