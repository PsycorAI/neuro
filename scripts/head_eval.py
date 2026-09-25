"""Evaluate the two auxiliary heads (ideas #3 and #6).

energy (#6):
  pair_acc      on held-out preference pairs, does E(chosen) < E(rejected)?
  bo_n          best-of-N reranking on held-out TOOL examples: sample N
                responses, pick argmin energy, and compare tool+arg exact-match
                against just taking the first sample. This is the number that
                says whether the verifier is worth the extra forward passes.

confidence (#3):
  reliability   predicted-confidence bucket -> actual accuracy. A useful head is
                monotone here; a useless one is flat.
  selective     refuse the lowest-confidence X% and report accuracy on the rest.
                That is the act/confirm/refuse routing decision, measured.

  CUDA_VISIBLE_DEVICES=1 python scripts/head_eval.py \
      --run opd_v3_363m --head energy --n 200
"""
import os, sys, glob, json, argparse
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import heads as H
import tool_data
from train_gpu import build_model
from train_sft import make_encoder_pair
from train_dpo import encode_pair, load_preferences_jsonl

PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"
TOOLS_DATA = os.path.join(ROOT, "training_data", "posttrain", "tools.jsonl")


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


@torch.no_grad()
def score_sequences(model, cap, head, ids, mask, device):
    cap.clear()
    model(ids.to(device))
    h = cap.hidden.float()
    return head(H.masked_mean_hidden(h, mask.to(device)))


@torch.no_grad()
def sample_response(model, prompt_ids, block_size, device, max_new, decode,
                    temperature):
    cur, gen = list(prompt_ids), []
    for _ in range(max_new):
        x = torch.tensor([cur[-block_size:]], dtype=torch.long, device=device)
        lg = model(x)[0, -1].float()
        lg[0] = float("-inf")
        if temperature > 0:
            tok = int(torch.multinomial(torch.softmax(lg / temperature, -1), 1))
        else:
            tok = int(lg.argmax())
        gen.append(tok); cur.append(tok)
        if tool_data.EOS_MARK in decode(gen[-4:]):
            break
    return gen, decode(gen)


def tool_exact(text, rec):
    mode, call, _ = tool_data.parse_response(text)
    if rec["target_mode"] == tool_data.MODE_TOOL:
        g = rec["target_call"]
        return bool(call and call.get("name") == g["name"]
                    and call.get("arguments") == g["arguments"])
    return mode == tool_data.MODE_ANSWER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_name holding the head + backbone")
    ap.add_argument("--head", required=True, choices=["energy", "confidence"])
    ap.add_argument("--backbone", default=None,
                    help="run_name of the frozen backbone (default: the head's own)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bo-n", type=int, default=4, help="candidates for best-of-N")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prefs", default=None, help="preference jsonl for pair_acc")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    head_path = os.path.join(ROOT, "models", "checkpoints", args.run,
                             f"{args.head}_head.pt")
    head, meta = H.load_head(head_path, args.device)
    backbone = args.backbone or meta["base"]
    ck = latest(backbone) if "/" not in backbone else backbone
    blob = torch.load(ck, map_location=args.device, weights_only=False)
    c = blob["cfg"]
    model = build_model(c).to(args.device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    encode, decode = make_encoder_pair(c)
    print(f"head: {head_path} (step {meta.get('step')})\nbackbone: {ck}")

    tools = tool_data.load_tools()
    test, _ = tool_data.load_tool_jsonl(TOOLS_DATA, tools, split="test")
    test = test[:args.n]

    with H.HiddenCapture(model) as cap:
        if args.head == "energy":
            # --- pair accuracy on held-out preference pairs ---------------
            if args.prefs:
                raw = load_preferences_jsonl(args.prefs)
                raw = [r for r in raw
                       if tool_data.split_of(r["prompt"]) == "test"][:args.n]
                ok = 0
                for r in raw:
                    w = encode_pair(r["prompt"], r["chosen"], encode, c["block_size"])
                    l = encode_pair(r["prompt"], r["rejected"], encode, c["block_size"])
                    if w is None or l is None:
                        continue
                    T = max(len(w[0]), len(l[0]))
                    ids = torch.zeros(2, T, dtype=torch.long)
                    msk = torch.zeros(2, T, dtype=torch.bool)
                    for i, (xx, mm) in enumerate((w, l)):
                        ids[i, :len(xx)] = torch.tensor(xx)
                        msk[i, :len(mm)] = torch.tensor(mm)
                    e = score_sequences(model, cap, head, ids, msk, args.device)
                    ok += int(e[0] < e[1])
                print(f"pair_acc  {ok/max(1,len(raw)):.3f}  (n={len(raw)}; 0.5 = useless)")

            # --- best-of-N reranking on tool examples ---------------------
            first_hit, best_hit = 0, 0
            for p in test:
                rec = p["rec"]
                pid = encode(p["prompt"])
                cands = []
                for k in range(args.bo_n):
                    gen, text = sample_response(
                        model, pid, c["block_size"], args.device, args.max_new,
                        decode, args.temperature)
                    ids = torch.tensor([pid + gen], dtype=torch.long)
                    msk = torch.zeros(1, ids.shape[1], dtype=torch.bool)
                    msk[0, len(pid):] = True
                    e = float(score_sequences(model, cap, head, ids, msk, args.device))
                    cands.append((e, text))
                first_hit += tool_exact(cands[0][1], rec)
                best_hit += tool_exact(min(cands)[1], rec)
            n = max(1, len(test))
            print(f"first_sample   {first_hit/n:.3f}")
            print(f"best_of_{args.bo_n}      {best_hit/n:.3f}   "
                  f"(gain {100*(best_hit-first_hit)/n:+.1f} pts)")

        else:
            # --- confidence: reliability + selective accuracy -------------
            rows = []
            for p in test:
                rec = p["rec"]
                pid = encode(p["prompt"])
                cap.clear()
                model(torch.tensor([pid], dtype=torch.long, device=args.device))
                conf = float(torch.sigmoid(head(cap.hidden.float()[:, -1])))
                _, text = sample_response(model, pid, c["block_size"], args.device,
                                          args.max_new, decode, 0.0)
                rows.append((conf, int(tool_exact(text, rec))))
            rows.sort(key=lambda r: r[0])
            n = len(rows)
            print(f"\noverall accuracy {sum(r[1] for r in rows)/n:.3f}  (n={n})")
            print("\nreliability (confidence bucket -> actual accuracy)")
            for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
                b = [r for r in rows if lo <= r[0] < lo + 0.2]
                if b:
                    print(f"  {lo:.1f}-{lo+0.2:.1f}: n={len(b):4d} acc={sum(x[1] for x in b)/len(b):.3f}")
            print("\nselective accuracy (refuse the least-confident X%)")
            for frac in (0.0, 0.1, 0.2, 0.3, 0.5):
                keep = rows[int(frac * n):]
                if keep:
                    print(f"  refuse {frac:.0%}: kept {len(keep):4d} "
                          f"acc={sum(x[1] for x in keep)/len(keep):.3f}")


if __name__ == "__main__":
    main()
