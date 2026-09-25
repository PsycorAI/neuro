"""Calibrated-abstention eval (idea #2).

The failure mode of abstention training is not "it doesn't abstain" -- it is
"it abstains at everything". So this measures BOTH directions and reports the
gap between them, not a single accuracy:

  idk_on_unknown   fraction of genuinely obscure questions answered with an
                   abstention                                    (want HIGH)
  idk_on_known     fraction of answerable questions answered with an abstention
                                                                 (want LOW)
  selectivity      idk_on_unknown - idk_on_known. A model that always abstains
                   scores 0 here, which is the point.
  answer_acc       on answerable questions, does the gold answer string appear
                   in the response (loose containment, so it is a floor not a
                   ceiling)

Data comes from the held-out split of abstain.jsonl (unanswerable half) and from
its answerable half, both keyed by the same deterministic hash split as the tool
eval so nothing leaks from training.

  CUDA_VISIBLE_DEVICES=1 python scripts/abstain_eval.py --run orpo_v3_363m --n 200
"""
import os, sys, glob, json, argparse
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tool_data import split_of
from train_gpu import build_model as _build
from train_sft import make_encoder_pair

DATA = os.path.join(ROOT, "training_data", "posttrain", "abstain.jsonl")
PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"
EOS_MARK = "\n###"

# Surface markers of an abstention. Deliberately simple and auditable: the
# generated IDK templates all contain one of these, and scoring by marker rather
# than by a classifier keeps the metric reproducible across runs.
IDK_MARKERS = ("i don't know", "i do not know", "i'm not able to answer",
               "i am not able to answer", "i can't answer", "i cannot answer",
               "i don't have a dependable answer", "genuinely don't know",
               "i couldn't say", "not sure about that", "no way of knowing",
               "outside what i can answer", "i'd need to look", "need a lookup",
               "would need to verify", "need to look it up")


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def is_abstention(text):
    t = text.lower()
    return any(m in t for m in IDK_MARKERS)


@torch.no_grad()
def generate(model, ids, block_size, device, max_new, decode):
    out = []
    cur = list(ids)
    for _ in range(max_new):
        x = torch.tensor([cur[-block_size:]], dtype=torch.long, device=device)
        tok = int(model(x)[0, -1].argmax())
        out.append(tok); cur.append(tok)
        if EOS_MARK in decode(out[-4:]):
            break
    return decode(out).split(EOS_MARK)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=200, help="per direction")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(DATA):
        print(f"no {DATA} yet -- generate the abstain target first"); return

    unknown, known = [], []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if split_of(r["prompt"]) != "test":
                continue
            (unknown if r.get("pair_type") == "unanswerable" else known).append(r)
    unknown, known = unknown[:args.n], known[:args.n]
    print(f"held-out: {len(unknown)} unanswerable, {len(known)} answerable")
    if not unknown or not known:
        print("need both directions in the test split -- rerun when generation "
              "has produced more data"); return

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    model = _build(c).to(args.device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    encode, decode = make_encoder_pair(c)
    print(f"{args.run}: step {blob.get('step')}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    res = {}
    for label, rows in (("unknown", unknown), ("known", known)):
        n_idk, n_hit, shown = 0, 0, 0
        for r in rows:
            ids = encode(PROMPT_FMT.format(message=r["prompt"]))
            text = generate(model, ids, c["block_size"], args.device,
                            args.max_new, decode)
            idk = is_abstention(text)
            n_idk += idk
            if label == "known" and r["chosen"].lower() in text.lower():
                n_hit += 1
            if shown < args.show:
                shown += 1
                print(f"  [{label}] {'IDK' if idk else 'ANS'} "
                      f"q={r['prompt'][:60]!r} -> {text[:80]!r}")
        res[label] = {"idk": n_idk / len(rows),
                      "acc": n_hit / len(rows) if label == "known" else None}

    sel = res["unknown"]["idk"] - res["known"]["idk"]
    print(f"\nidk_on_unknown  {res['unknown']['idk']:.3f}   (want HIGH)")
    print(f"idk_on_known    {res['known']['idk']:.3f}   (want LOW)")
    print(f"selectivity     {sel:+.3f}   (always-abstain scores 0.000)")
    print(f"answer_acc      {res['known']['acc']:.3f}   (gold string contained; a floor)")


if __name__ == "__main__":
    main()
