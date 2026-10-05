"""nl_fact_eval.py -- 2026-10-04 (r25). Natural-language fact recall, inside vs beyond the attention window (the demo task:
"tell it a fact, ask later").
Episode: K facts "{Name}'s favorite {cat} is {val}." (K distinct names, distinct values, one category) + real val text
+ "Question: What is {Name_q}'s favorite {cat}? Answer: {Name_q}'s favorite {cat} is" -> compare the logits of the first
token of every value of that category (among-candidates accuracy; chance = 1/len(values); with K facts in context,
copying a random in-context value scores 1/K).
Placements: in@g = everything inside one attention chunk with g filler tokens before the question;
            x@o  = facts at the start of chunk A, question starting at offset o of chunk B (only the memory can carry it).
Memory-ablated accuracy is reported for x@0. --chunk overrides the model's attn_chunk (default 512 if the model has none).
  python temp/nl_fact_eval.py [--chunk 512] [--K 1,4] name=ckpt.pt [...]"""
import sys, argparse, random, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.path.insert(0, "/home/glenn/projects/neuro/scripts")
from train_gpu import build_model, StreamText, BIN
from chat import load_tokenizer
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--K", default="1,4")
ap.add_argument("--n", type=int, default=192)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()

CATS = {
    "color": "red blue green yellow purple orange pink brown black white gray silver",
    "animal": "dog cat horse rabbit tiger lion bear wolf fox eagle shark snake",
    "fruit": "apple banana cherry grape lemon mango peach pear plum melon kiwi lime",
    "city": "Paris London Tokyo Rome Berlin Madrid Cairo Sydney Moscow Dublin Vienna Boston",
    "sport": "soccer tennis golf hockey rugby baseball cricket boxing swimming skiing cycling chess",
}
NAMES = ("Emma Liam Olivia Noah Ava Mason Sophia Lucas Mia Ethan Lily James Grace Henry Chloe Jack Zoe Owen Ella Leo "
         "Ruby Max Anna Sam Nora Ben Sara Tom Kate Dan Amy Paul Jane Mark Lucy Adam Rose Carl Ivy Hugo").split()


def episodes(enc, K, rng):
    """Yield (fact_ids, question_ids, cand_first_ids, target_index)."""
    cats = {}
    for cat, vs in CATS.items():
        firsts, keep = [], []
        for v in vs.split():
            t = enc(" " + v)
            if t and t[0] > 0 and t[0] not in firsts:
                firsts.append(t[0]); keep.append(v)
        cats[cat] = (keep, firsts)
    names = [n for n in NAMES if all(t > 0 for t in enc(" " + n))]
    while True:
        cat = rng.choice(sorted(cats)); vals, firsts = cats[cat]
        who = rng.sample(names, K); what = rng.sample(range(len(vals)), K)
        facts = " ".join(f"{n}'s favorite {cat} is {vals[i]}." for n, i in zip(who, what))
        q = rng.randrange(K)
        question = f" Question: What is {who[q]}'s favorite {cat}? Answer: {who[q]}'s favorite {cat} is"
        yield enc(facts), enc(question), firsts, what[q]


def build(ep, total_to_q, data):
    """facts + filler (real val text) so the question starts at index total_to_q."""
    f, qn, _, _ = ep
    g = total_to_q - len(f)
    assert g >= 0
    filler = data.batch(1, g, "val", "cpu")[0][0].tolist() if g > 0 else []
    return f + filler + qn


for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = a.chunk or int(c.get("attn_chunk", 0) or 0) or 512
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    if c.get("lut_subword_fallback", False):
        import lut_spell
        enc = lut_spell.make_encoder(c["vocab"], subword_fallback=True)
    else:
        enc, _ = load_tokenizer(c["vocab"])
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"))
    out = []
    for K in [int(k) for k in a.K.split(",")]:
        # (label, question start index, ablate)
        settings = [("in@0", None, False), ("in@200", 200, False), ("x@0", C, False), ("x@200", C + 200, False),
                    ("x@0 off", C, True)]
        for label, qpos, abl in settings:
            rng = random.Random(1234 + K); np.random.seed(1234 + K)
            gen = episodes(enc, K, rng)
            hit = 0
            for _ in range(a.n // 32):
                eps = [next(gen) for _ in range(32)]
                if qpos is None:
                    seqs = [e[0] + e[1] for e in eps]
                elif label.startswith("in"):
                    seqs = [build(e, len(e[0]) + qpos, data) for e in eps]
                else:
                    seqs = [build(e, qpos, data) for e in eps]
                L = max(len(sq) for sq in seqs)
                if label.startswith("in"):
                    assert L <= C, f"in-window episode too long ({L} > {C})"
                # left-align, right-pad with 0; read each row at its own last real position (causal -> pads unseen)
                x = torch.zeros(len(seqs), L, dtype=torch.long)
                for i, sq in enumerate(seqs):
                    x[i, :len(sq)] = torch.tensor(sq)
                last = torch.tensor([len(sq) - 1 for sq in seqs])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(x.cuda(), ablate_memory=abl).float()[torch.arange(len(seqs)), last.cuda()].cpu()
                for e, row in zip(eps, lg):
                    hit += int(row[torch.tensor(e[2])].argmax().item() == e[3])
            out.append(f"K{K} {label} {100.0 * hit / (a.n // 32 * 32):5.1f}")
    print(f"{name:26s} C={C} | " + " | ".join(out), flush=True)
    del m; torch.cuda.empty_cache()
print("chance = 1/12 = 8.3%; copying a random in-context value = 100/K %")
