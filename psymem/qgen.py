# psymem/qgen.py -- 2026-10-27 (Neuro 85): synthetic (question, memory turn) pairs to train the Psymem spiking encoder.
#   recall   : a later question that would find one turn of data/psymem_enc/corpus_400000.jsonl (tulu-3 chats), as a person
#              remembers it (paraphrased, vague), greedy;
#   personal : a casual message sharing a personal detail (random speaker / topic / date) + a later question about it, sampled.
# Generator = Llama-3.2-3B-Instruct. No LoCoMo / LongMemEval text. Appends in chunks, so a restart resumes.
# Usage: python psymem/qgen.py recall|personal N [chunk]   -> data/psymem_enc/pairs_{kind}.jsonl  {"q", "pos", "kind"}
import sys, os, json, random, re
import qa

OUT = '/home/glenn/projects/neuro/data/psymem_enc/'
RECALL = ("Here is one message from a chat that happened weeks ago:\n\n{t}\n\nWrite ONE question that someone might ask now to "
          "find this message again, the way people actually remember things: paraphrase, use different words from the "
          "message, and do not include the answer. Output only the question.")
PERSONAL = ("Write a short, casual chat message (1-3 sentences) in which {who} mentions a specific personal detail about {topic}, "
            "with concrete specifics (a name, place, number or date){when}. Then write the question a friend might naturally ask "
            "weeks later to recall that detail, for example 'What guitar did Hannah buy?' or 'Where did I go hiking in May?'. "
            "Name the subject plainly, but do not include the answer.\nFormat exactly:\n"
            "MESSAGE: <the message>\nQUESTION: <the question>")
TOPICS = ('their pet', 'a sibling', 'their parents', 'their kids', 'their partner', 'their job', 'a coworker', 'their boss',
          'the commute', 'their apartment', 'a neighbor', 'the garden', 'a recipe they cooked', 'a favorite restaurant',
          'a diet change', 'an allergy', 'a workout routine', 'a running race', 'yoga class', 'a sports team',
          'a musical instrument', 'a concert', 'a book they finished', 'a TV show', 'a movie', 'a video game', 'a board game',
          'a craft hobby', 'painting', 'photography', 'a past trip', 'vacation plans', 'their car', 'a new bike', 'a phone',
          'a laptop purchase', 'a budget goal', 'savings', 'rent', 'moving house', 'a class they take', 'their degree',
          'an exam', 'a certification', 'a doctor visit', 'a medication', 'sleep habits', 'a birthday', 'an anniversary',
          'a wedding', 'a holiday tradition', 'volunteering', 'a charity event', 'learning a language', "a friend's news",
          'a festival', 'camping', 'a hike', 'a beach day', 'a museum visit', 'coffee', 'tea', 'houseplants',
          'a smart home device', 'a subscription', 'a home repair', 'a gift they bought', 'a job interview', 'a promotion')
NAMES = ('I', 'I', 'I', 'Maya', 'Leo', 'Priya', 'Sam', 'Jordan', 'Aisha', 'Tom', 'Elena', 'Kenji', 'Grace', 'Omar', 'Lucia',
         'Ben', 'Zoe', 'Ravi', 'Hannah', 'Diego', 'Nora', 'Kwame', 'Ivy', 'Felix')


def recall_items(n):
    texts = [json.loads(l) for l in open(OUT + 'corpus_400000.jsonl')]
    texts = [t for t in texts if '```' not in t and '\\(' not in t and '$$' not in t and '=' not in t  # skip code / maths
             and len(re.findall(r'\d', t)) < 15 and not re.search(r'\b(calculate|solve|find the|how many)\b', t.lower())
             and sum(c.isascii() for c in t) > 0.95 * len(t)]  # English-ish only (the front end is an English embedder)
    random.Random(1).shuffle(texts)
    return [(RECALL.format(t=t[:1200]), t) for t in texts[:n]]


def personal_items(n):
    rng = random.Random(2)
    out = []
    for _ in range(n):
        who, topic = rng.choice(NAMES), rng.choice(TOPICS)
        when = f', and say it happened on {rng.randint(1, 28)} {rng.choice("January February March April May June July August September October November December".split())} {rng.randint(2021, 2024)}' if rng.random() < 0.3 else ''
        out.append((PERSONAL.format(who='the speaker (writing as "I")' if who == 'I' else who, topic=topic, when=when), who))
    return out


if __name__ == '__main__':
    kind, N = sys.argv[1], int(sys.argv[2])
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    path = OUT + f'pairs_{kind}.jsonl'
    done = int(open(path + '.done').read()) if os.path.exists(path + '.done') else 0  # prompts finished (rows are filtered)
    items = (recall_items if kind == 'recall' else personal_items)(N)
    budget = int(os.environ.get('QA_BUDGET', 24000))
    for i in range(done, N, chunk):
        part = items[i:i + chunk]
        kw = dict(do_sample=True, temperature=0.9, top_p=0.95) if kind == 'personal' else {}
        outs = qa.generate(qa.READER, [[{'role': 'user', 'content': p}] for p, _ in part], 120 if kind == 'personal' else 48, budget, **kw)
        with open(path, 'a') as f:
            for (p, x), o in zip(part, outs):
                if kind == 'recall':
                    q = o.strip().strip('"').split('\n')[0].strip()
                    if q.endswith('?') and 10 < len(q) < 300:
                        f.write(json.dumps(dict(q=q, pos=x, kind=kind)) + '\n')
                else:
                    m = re.search(r'MESSAGE:\s*(.+?)\s*QUESTION:\s*(.+?\?)', o, re.S)
                    if m:
                        speaker = 'user' if x == 'I' else x
                        f.write(json.dumps(dict(q=m[2].strip().strip('"'), pos=f'{speaker}: {m[1].strip().strip(chr(34))}',
                                                kind=kind)) + '\n')
        open(path + '.done', 'w').write(str(min(i + chunk, N)))
        print(f'{kind}: {min(i + chunk, N)}/{N} prompts done', flush=True)
