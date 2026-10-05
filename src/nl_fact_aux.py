"""nl_fact_aux.py -- r25 (2026-10-05, default off). Synthetic natural-language fact episodes: the "tell it a fact, ask
later" skill the demo needs (temp/nl_fact_eval.py showed 150M models at chance beyond the attention window, and weak even
inside it). An episode = K facts ("Alice lives in Paris.") + real train text + a question about one fact; the loss is CE
on the first token of the answer at the question's last position (like the MQAR aux). With attn_chunk C, a fraction
(1 - in_frac) of episodes put the facts at the start of chunk A and the question in chunk B, so only the memory can
carry the fact; the rest stay inside one chunk (attention's job).
HELD OUT for the eval: the eval's names and its "favorite {category}" template never appear here (asserted below)."""
import random
import numpy as np
import torch

TRAIN_NAMES = ("Alice Bob Carol David Eve Frank George Hannah Isaac Julia Kevin Laura Michael Nina Oscar Peter Rachel "
               "Steve Tina Victor Wendy Aaron Bella Caleb Diana Eric Fiona Helen Ian Kyle Leah Martin Naomi Oliver Paula "
               "Ryan Tyler Vera Walter Andrew Brian Claire Derek Ellen Felix Gloria Harry Irene Jason Karen Louis Monica "
               "Nathan Olga Patrick Robert Sharon Thomas Ursula Vincent Bruce Cindy Dennis Elena Gary Heidi Jerry Kim "
               "Larry Megan Neil Pamela Roger Sandra Teresa Wayne Albert Betty Chris Donna Edward Frances Gordon Holly "
               "Jacob Linda Matthew Nancy Philip Ruth Simon Theresa Arthur Barbara Charles Dorothy Eugene Gina Howard "
               "Joan Keith Lisa Marcus Nicole Raymond Susan Timothy Victoria").split()
EVAL_NAMES = ("Emma Liam Olivia Noah Ava Mason Sophia Lucas Mia Ethan Lily James Grace Henry Chloe Jack Zoe Owen Ella "
              "Leo Ruby Max Anna Sam Nora Ben Sara Tom Kate Dan Amy Paul Jane Mark Lucy Adam Rose Carl Ivy Hugo").split()
assert not set(TRAIN_NAMES) & set(EVAL_NAMES)

# relation: (fact, question repeating the fact's stem, paraphrased question answered right after "Answer:", values)
RELATIONS = [
    ("{n} lives in {v}.", " Question: Where does {n} live? Answer: {n} lives in",
     " Question: Which city is home to {n}? Answer:",
     "Paris London Tokyo Rome Berlin Madrid Cairo Sydney Moscow Dublin Vienna Boston Chicago Toronto Seattle Denver "
     "Miami Dallas Atlanta Houston Lisbon Prague Warsaw Athens Oslo Stockholm Helsinki Delhi Mumbai Beijing Seoul Lima"),
    ("{n} has a pet {v}.", " Question: What pet does {n} have? Answer: {n} has a pet",
     " Question: Which animal does {n} keep at home? Answer: a",
     "dog cat horse rabbit hamster parrot turtle snake goat pig duck lizard mouse frog fish pony"),
    ("{n} works as a {v}.", " Question: What is {n}'s job? Answer: {n} works as a",
     " Question: What does {n} do for a living? Answer: a",
     "teacher doctor nurse farmer pilot baker lawyer driver painter writer singer dancer chef soldier plumber judge "
     "banker carpenter dentist sailor"),
    ("{n} was born in {v}.", " Question: When was {n} born? Answer: {n} was born in",
     " Question: In which month was {n} born? Answer:",
     "January February March April May June July August September October November December"),
    ("The secret code of {n} is {v}.", " Question: What is the secret code of {n}? Answer: The secret code of {n} is",
     " Question: Which code belongs to {n}? Answer:",
     " ".join(str(i) for i in range(100, 1000, 7))),
    ("{n} plays the {v}.", " Question: Which instrument does {n} play? Answer: {n} plays the",
     " Question: What instrument is {n}'s? Answer: the",
     "piano guitar violin drums flute trumpet cello harp saxophone clarinet organ banjo"),
    ("{n} drives a {v} car.", " Question: What color is {n}'s car? Answer: {n} drives a",
     " Question: Which color car does {n} own? Answer:",
     "red blue green yellow purple orange pink brown black white gray silver"),
]
for _f, _q, _p, _ in RELATIONS:
    assert "favorite" not in _f + _q + _p


class NLFactAux:
    """Bank of single facts tokenized once; episodes are assembled per batch from K facts with distinct names."""

    def __init__(self, vocab, chunk, seed=0, k_max=4, in_frac=0.3, paraphrase=0.5, n_bank=20000,
                 subword_fallback=False):
        import sys
        sys.path.insert(0, "/home/glenn/projects/neuro/scripts")
        if subword_fallback:                 # must match how the corpus was tokenized (config lut_subword_fallback)
            import lut_spell
            enc = lut_spell.make_encoder(vocab, subword_fallback=True)
        else:
            from chat import load_tokenizer
            enc, _ = load_tokenizer(vocab)
        self.C, self.k_max, self.in_frac = int(chunk), int(k_max), float(in_frac)
        self.rng = random.Random(1_000_003 * (seed + 1))
        name_ok = [n for n in TRAIN_NAMES if all(t > 0 for t in enc(" " + n))]
        space = set(enc(" "))
        bank = []
        for _ in range(n_bank):
            f, q, p, vals = self.rng.choice(RELATIONS)
            n = self.rng.choice(name_ok); v = self.rng.choice(vals.split())
            para = self.rng.random() < paraphrase
            fi, qi, ai = enc(" " + f.format(n=n, v=v)), enc((p if para else q).format(n=n)), enc(" " + v)
            while len(ai) > 1 and ai[0] in space:   # Llama-3 splits " 107" -> " ", "107": the space joins the question
                qi.append(ai.pop(0))
            if min(fi + qi + ai) > 0:
                bank.append((n, fi, qi, ai[0]))
        self.bank = bank
        print(f"nl_fact_aux: {len(bank)} facts, {len(name_ok)} names, {len(RELATIONS)} relations, chunk {self.C}, "
              f"in-chunk {self.in_frac:.0%}")

    def batch(self, B, data, device):
        """Returns x (B, 2C) right-padded with 0, last (B,) index of each question's final token, target (B,)."""
        C, r = self.C, self.rng
        x = torch.zeros(B, 2 * C, dtype=torch.long)
        last = torch.zeros(B, dtype=torch.long); tgt = torch.zeros(B, dtype=torch.long)
        for i in range(B):
            K = r.randint(1, self.k_max)
            facts, names = [], set()
            while len(facts) < K:
                e = r.choice(self.bank)
                if e[0] not in names:
                    names.add(e[0]); facts.append(e)
            fids = [t for e in facts for t in e[1]]
            q = r.choice(facts)
            if r.random() < self.in_frac:                         # inside chunk A (attention can see the facts)
                qstart = len(fids) + r.randint(0, max(0, C - len(fids) - len(q[2]) - 1))
            else:                                                 # question in chunk B (memory only)
                qstart = C + r.randint(0, C - len(q[2]) - 1)
            g = qstart - len(fids)
            s = r.randint(0, data.split - g - 1)
            filler = data.lut[np.asarray(data.data[s:s + g], dtype=np.int64)].tolist() if g > 0 else []
            seq = fids + filler + q[2]
            x[i, :len(seq)] = torch.tensor(seq)
            last[i] = len(seq) - 1; tgt[i] = q[3]
        return x.to(device), last.to(device), tgt.to(device)
