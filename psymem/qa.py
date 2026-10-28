# psymem/qa.py -- 2026-10-27 (Neuro 85): end-to-end QA (memory product plan step 6). Each system retrieves its top-k turns;
# a fixed local reader answers from them (shown in date order); scored by token F1 (SQuAD-style) and a fixed local LLM judge
# (yes / no; prompts adapted from LongMemEval's judge). Same reader, judge, k and prompt for every system.
# LoCoMo category 5 (adversarial) is left out here, as in the Mem0 paper; abstention is tested separately.
# Usage: python psymem/qa.py locomo|lme K systems...   -> table on stdout + logs/n85_qa_{which}_k{K}.jsonl
import sys, os, json, re, time, string, collections, gc
import numpy as np, torch
import bench
from transformers import AutoTokenizer, AutoModelForCausalLM

READER, JUDGE = 'meta-llama/Llama-3.2-3B-Instruct', 'microsoft/Phi-3.5-mini-instruct'
SYS = ("You answer questions about a user's past conversations, using only the retrieved memories below. "
       "Reply with a short answer (a phrase or one sentence). If the memories do not contain the answer, give your best guess. "
       "For questions about when something happened, give a calendar date, worked out from the memory's date if the text "
       "says 'yesterday', 'last week' and so on.")
JUDGE_BASE = ("I will give you a question, a correct answer, and a response from a model. Answer yes if the response contains "
              "the correct answer or is equivalent to it. If the response only contains a subset of the information the "
              "answer requires, answer no.{extra}\n\nQuestion: {q}\n\nCorrect answer: {a}\n\nModel response: {r}\n\n"
              "Is the model response correct? Answer yes or no only.")
JUDGE_EXTRA = {
    'temporal-reasoning': " Do not penalize off-by-one errors in a number of days, weeks or months.",
    'knowledge-update': " If the response mentions older information along with the updated answer, it is correct as long as "
                        "the updated answer is the required one.",
    'single-session-preference': " The correct answer here is a rubric for a personalized response: answer yes if the response "
                                 "recalls and uses the user's personal information correctly; it need not cover every point.",
}


def f1(pred, gold):
    def toks(s):
        s = ''.join(ch for ch in s.lower() if ch not in string.punctuation)
        return re.sub(r'\b(a|an|the|and)\b', ' ', s).split()
    p, g = toks(pred), toks(gold)
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    return 0. if not same else 2 * same / (len(p) + len(g))


@torch.no_grad()
def generate(path, chats, max_new, budget, **gen_kw):
    """Chat completion for many prompts (greedy unless gen_kw says otherwise); batches sorted by length under a token budget."""
    gen_kw = dict(do_sample=False) | gen_kw
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = 'left'
    tok.pad_token = tok.pad_token or tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    ids = [list(tok.apply_chat_template(c, add_generation_prompt=True, return_dict=True)['input_ids']) for c in chats]
    order, out, i = sorted(range(len(ids)), key=lambda j: len(ids[j])), [None] * len(ids), 0
    while i < len(order):
        longest = len(ids[order[min(i + 63, len(order) - 1)]])  # sorted, so this bounds the batch
        b = order[i:i + max(1, min(64, budget // (longest + max_new)))]
        L = max(len(ids[j]) for j in b)
        x = torch.tensor([[tok.pad_token_id] * (L - len(ids[j])) + ids[j] for j in b]).cuda()
        mask = torch.tensor([[0] * (L - len(ids[j])) + [1] * len(ids[j]) for j in b]).cuda()
        y = m.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **gen_kw)
        for j, row in zip(b, y[:, L:]):
            out[j] = tok.decode(row, skip_special_tokens=True).strip()
        i += len(b)
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    which, K, systems = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    t0 = time.time()
    mems = bench.load_locomo() if which == 'locomo' else bench.load_lme()
    mems = mems[:int(os.environ.get('QA_MEMS', len(mems)))]  # QA_MEMS=1 for a smoke test
    emb = bench.Embedder(cache_file=bench.D + f'emb_{which}.pkl')
    bench.SCORERS.update({s: bench.psy_scorer(s) for s in systems if s not in bench.SCORERS})
    rows = []  # one per (system, question)
    for m in mems:
        c = bench.ctx_base(m, emb)
        for q, gold, cat, ans, c['now'] in m['qs']:
            qdate = c['now']
            if cat == 5:
                continue
            for s in systems:
                top = sorted(np.argsort(-bench.SCORERS[s](c, q))[:K])  # date order = diary order
                mem = '\n'.join(f'[{n + 1}] ({m["items"][i][2]}) {m["items"][i][1]}' for n, i in enumerate(top))
                user = f'Memories:\n{mem}\n\n' + (f'Current date: {qdate}\n' if qdate else '') + f'Question: {q}\nShort answer:'
                rows.append(dict(sys=s, cat=cat, q=q, ans=ans, chat=[{'role': 'system', 'content': SYS},
                                                                    {'role': 'user', 'content': user}]))
    budget = int(os.environ.get('QA_BUDGET', 24000))  # tokens per batch; 48k filled the 5090 on LongMemEval's long prompts
    for r, a in zip(rows, generate(READER, [r['chat'] for r in rows], 64, budget)):
        r['pred'] = a; r['f1'] = f1(a, r['ans'])
    jc = [[{'role': 'user', 'content': JUDGE_BASE.format(extra=JUDGE_EXTRA.get(r['cat'], ''), q=r['q'], a=r['ans'], r=r['pred'])}]
          for r in rows]
    for r, v in zip(rows, generate(JUDGE, jc, 4, budget)):
        r['judge'] = float(v.lower().startswith('yes'))
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for key in (r['cat'], 'ALL'):
            res[(r['sys'], key)]['f1'].append(r['f1']); res[(r['sys'], key)]['judge'].append(r['judge'])
    cats = ['ALL'] + sorted({r['cat'] for r in rows}, key=str)
    print(f'== QA {which} k={K}: {len(rows) // len(systems)} questions, {time.time() - t0:.0f} s')
    for mt in ('judge', 'f1'):
        print(f'-- {mt}'.ljust(44) + ''.join(f'{str(c)[:9]:>10}' for c in cats))
        for s in systems:
            print(f'{s[:44]:<44}' + ''.join(f"{100 * np.mean(res[(s, c)][mt]):>10.1f}" for c in cats))
    with open(f'/home/glenn/projects/neuro/logs/n85_qa_{which}_k{K}.jsonl', 'w') as fo:
        for r in rows:
            fo.write(json.dumps({k: v for k, v in r.items() if k != 'chat'}) + '\n')
