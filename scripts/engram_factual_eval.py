"""Engram factual-recall eval — the instrument v1 was missing.

Why this exists
---------------
Neuro v1 Engram closed inconclusive-to-negative gated on general val_ppl. But
DeepSeek V4's own ablation shows general capability is nearly flat with/without
Engram, while FACTUAL benchmarks collapse to 29-44% (TriviaQA→29%). We were
measuring the wrong thing: v1's memory-Δ was consistently +13-74 (the pathway
was firing), general ppl just can't see it.

This script is the missing signal. It runs LAMA-style cloze completions on a
small curated fact set, twice — once with Engram enabled, once with it ablated
(via the existing `ablate_engram=True` forward-pass flag) — and reports the gap.
The gap IS the answer to "is our Engram doing anything for facts?" A model with
no Engram will show identical numbers in both modes; a working Engram will show
a real drop on ablation.

Scoring (teacher-forced, supports multi-token targets)
-------
For a fact with prompt of P subvocab tokens and target of K tokens:
  1. Concatenate prompt+target, feed through the model.
  2. Position P-1 predicts target[0], position P predicts target[1], ...
  3. Sum log P(target[k]) across all K positions -> total log-prob of target.
  4. rank / top-1 / top-10 are read from the FIRST target position (P-1) --
     the single position most-informative about factual retrieval; trailing
     positions get an easier task once the head is set.

Metrics (higher log-prob / top-K = better; lower rank = better):
  sum log-prob : log P(full target | prompt), teacher-forced
  top-1, top-10: hit rate on target[0] (first-token-only, matched across facts)
  rank         : 1-indexed rank of target[0]

Applies to

Applies to
----------
Any Neuro checkpoint using the vocab=16384 Llama-3 subvocab (screen/30M/90M/
350M runs). NOT for the 500M v2 (Phi-3.5 vocab=32064) — that would need a
separate LUT path. For v2 support once wired, add an `--engram_which v2` flag
and a matching model.forward ablation kwarg.

Usage
-----
  python scripts/engram_factual_eval.py --run screen_engram_30m
  python scripts/engram_factual_eval.py --run <name> --facts my_facts.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
LUT_PATH = os.path.join(ROOT, "data", "vocab_map_16384.npy")


# Curated fact set: 50 diverse LAMA-style cloze prompts. Common single-token
# answers preferred so most survive the 16k Llama-3 subvocab. Format: each is
# (prompt_without_answer, target_word_with_leading_space). Leading space is
# critical -- BPE tokenizes " Paris" and "Paris" as different first tokens,
# and next-token prediction after "is" wants the space-prefixed variant.
FACTS = [
    # Capitals (very common; should survive subvocab)
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of Russia is", " Moscow"),
    ("The capital of China is", " Beijing"),
    ("The capital of Canada is", " Ottawa"),
    ("The capital of Egypt is", " Cairo"),
    ("The capital of Australia is", " Canberra"),
    # Countries / continents
    ("Paris is a city in", " France"),
    ("Berlin is a city in", " Germany"),
    ("Tokyo is a city in", " Japan"),
    ("Cairo is a city in", " Egypt"),
    ("Mount Everest is located in", " Nepal"),
    ("The Great Wall is in", " China"),
    ("The Eiffel Tower is in", " Paris"),
    ("The Amazon River flows through", " Brazil"),
    # Language / nationality
    ("The main language of France is", " French"),
    ("The main language of Germany is", " German"),
    ("The main language of Japan is", " Japanese"),
    ("The main language of Spain is", " Spanish"),
    ("The main language of Brazil is", " Portuguese"),
    ("The main language of Egypt is", " Arabic"),
    # Currency
    ("The currency of Japan is the", " yen"),
    ("The currency of the United Kingdom is the", " pound"),
    # Science / people
    ("Water is composed of hydrogen and", " oxygen"),
    ("The chemical symbol for gold is", " Au"),
    ("The chemical symbol for iron is", " Fe"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The closest star to Earth is the", " Sun"),
    ("Albert Einstein was born in", " Germany"),
    ("Isaac Newton was born in", " England"),
    ("Leonardo da Vinci was born in", " Italy"),
    ("Marie Curie was born in", " Poland"),
    ("Charles Darwin was born in", " England"),
    # Culture / literature
    ("The author of Hamlet is William", " Shakespeare"),
    ("The author of The Odyssey is", " Homer"),
    ("The author of Don Quixote is", " Cervantes"),
    # Geography
    ("The longest river in the world is the", " Nile"),
    ("The largest ocean is the", " Pacific"),
    ("The tallest mountain is Mount", " Everest"),
    ("The Sahara is a desert in", " Africa"),
    # Institutions / cities of...
    ("Hollywood is famous for", " movies"),
    ("The Louvre museum is in", " Paris"),
    ("The Vatican is in", " Rome"),
    ("Wall Street is in", " New"),
    # Simple relations
    ("The sun rises in the", " east"),
    ("The opposite of hot is", " cold"),
    ("The opposite of day is", " night"),
]


def latest_ckpt(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    # Delegate to train_gpu.build_model -- single source of truth for every
    # kwarg. The hand-rolled copy this used to be drifted (missing gdn2_enabled,
    # use_rmsnorm, embed_scale, mtp_*), which silently constructed the wrong
    # architecture for Batch A / GDN-2 checkpoints and made the ON-vs-OFF
    # verdict meaningless.
    from train_gpu import build_model
    return build_model(c).to(device)


def load_encoder():
    """Return a `encode(text) -> LongTensor(1, T)` mapping to the 16k subvocab.
    OOV Llama-3 tokens (subvocab id 0) are dropped, matching scripts/generate.py.
    """
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut = np.load(LUT_PATH)

    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        mapped = [int(lut[t]) for t in ids if lut[t] != 0]
        return mapped

    return encode, tk, lut


@torch.no_grad()
def score(model, block_size, prompt_ids, target_ids, device):
    """Teacher-forced multi-token target scoring, with Engram enabled AND ablated.

    Returns dict of both, or None if prompt+target got left-truncated so hard
    that no prompt tokens remain to condition on.
    """
    full = prompt_ids + target_ids
    K = len(target_ids)
    # Left-truncate to fit block_size. Prompt tokens are dropped first (Engram
    # only uses the last N tokens anyway); we keep at least 1 prompt token so
    # target[0] has something to be predicted from.
    if len(full) > block_size:
        full = full[-block_size:]
    P = len(full) - K
    if P < 1:
        return None
    ids = torch.tensor([full], dtype=torch.long, device=device)
    tgts = torch.tensor(target_ids, device=device)
    pred_pos = torch.arange(P - 1, P - 1 + K, device=device)       # (K,)
    out = {}
    for tag, ablate in (("on", False), ("off", True)):
        logits = model(ids, ablate_engram=ablate, ablate_engram_v2=ablate)[0]   # (T, V)
        logp = F.log_softmax(logits[pred_pos], dim=-1)             # (K, V)
        tgt_logp = float(logp.gather(1, tgts[:, None]).squeeze(1).sum().item())
        first = logits[P - 1]                                       # first-target-position logits
        rank = int((first > first[target_ids[0]]).sum().item()) + 1
        top1 = int(first.argmax().item() == target_ids[0])
        top10 = int(target_ids[0] in torch.topk(first, 10).indices.tolist())
        out[tag] = dict(logp=tgt_logp, rank=rank, top1=top1, top10=top10)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint dir under models/checkpoints/")
    ap.add_argument("--facts", default=None,
                    help="optional JSON: list of [prompt, target] pairs; default = built-in curated set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"loading checkpoint: {args.run}")
    blob = torch.load(latest_ckpt(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    print(f"  step={blob.get('step','?')} tokens={blob.get('tokens', 0)/1e6:.0f}M "
          f"vocab={c['vocab']} engram_enabled={c.get('engram_enabled', False)} "
          f"engram_v2_enabled={c.get('engram_v2_enabled', False)}")

    if c["vocab"] != 16384:
        print(f"[WARN] this eval assumes 16k Llama-3 subvocab; got vocab={c['vocab']}. "
              f"Continuing anyway — token IDs may be wrong.")

    m = build(c, args.device)
    miss, unexp = m.load_state_dict(blob["model"], strict=False)
    dropped = [k for k in unexp if "lif." not in k]
    if dropped:
        print(f"[WARN] dropped keys: {dropped[:3]}{'...' if len(dropped) > 3 else ''}")
    m.eval()
    engram_wired = c.get("engram_enabled", False) or c.get("engram_v2_enabled", False)
    if not engram_wired:
        print("[NOTE] this checkpoint has engram_enabled=False and engram_v2_enabled=False; "
              "enabled/ablated numbers should be identical (sanity check for the harness).")

    facts = FACTS
    if args.facts:
        facts = [tuple(x) for x in json.load(open(args.facts))]
    print(f"fact set: {len(facts)}")

    encode, _, _ = load_encoder()
    block_size = c.get("block_size", 512)

    on = dict(logp=[], rank=[], top1=[], top10=[])
    off = dict(logp=[], rank=[], top1=[], top10=[])
    skipped = []

    for prompt, target in facts:
        prompt_ids = encode(prompt)
        target_ids = encode(target)
        if not prompt_ids:
            skipped.append((prompt, "prompt all-OOV")); continue
        if not target_ids:
            skipped.append((prompt, "target all-OOV")); continue
        r = score(m, block_size, prompt_ids, target_ids, args.device)
        if r is None:
            skipped.append((prompt, "prompt+target > block_size")); continue
        for k in on: on[k].append(r["on"][k])
        for k in off: off[k].append(r["off"][k])

    n = len(on["logp"])
    if n == 0:
        print("[ERR] no facts scored — all were skipped. Check LUT / fact set.")
        return

    def mean(xs): return sum(xs) / len(xs)

    print(f"\nscored {n}/{len(facts)} facts ({len(skipped)} skipped)")
    print(f"{'':>18}  {'ENGRAM ON':>12}   {'ENGRAM OFF':>12}   {'Δ (on-off)':>12}")
    for k, label, fmt in [
        ("logp",  "mean log-prob ", "{:>12.3f}"),
        ("rank",  "mean rank     ", "{:>12.1f}"),
        ("top1",  "top-1 acc %   ", "{:>11.1f}%"),
        ("top10", "top-10 acc %  ", "{:>11.1f}%"),
    ]:
        scale = 100 if k.startswith("top") else 1
        a = mean(on[k]) * scale
        b = mean(off[k]) * scale
        d = a - b
        # For "rank", lower is better -> report ON-OFF as negative-is-good.
        print(f"{label:>18}  " + fmt.format(a) + "   " + fmt.format(b) + "   " + fmt.format(d))

    print("\nInterpretation:")
    print("  Engram working  = ON meaningfully better than OFF")
    print("                    (higher log-prob, lower rank, higher top-K).")
    print("  Engram inactive = ON and OFF nearly identical (path exists but contributes ~0).")
    print("  Engram harmful  = ON worse than OFF (rare; possible if mis-trained).")

    if skipped and len(skipped) <= 12:
        print(f"\nSkipped facts:")
        for p, why in skipped:
            print(f"  [{why}] {p!r}")


if __name__ == "__main__":
    main()
