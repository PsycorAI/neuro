"""Tool-calling eval for a post-trained checkpoint (idea #4).

Measures, on a held-out split of training_data/posttrain/tools.jsonl, both with
and without grammar-constrained decoding:

  parse       fraction of responses that parse as a well-formed call/answer
  mode_acc    TOOL vs ANSWER chosen correctly
  tool_acc    correct tool named (over gold-TOOL examples)
  args_exact  every argument exactly right (over gold-TOOL examples)
  halluc      named a tool that was NOT in the offered list  <- the negative
              class that irrelevant-tool training is supposed to fix
  refuse_acc  gold-ANSWER examples answered in ANSWER mode (no needless call)

The constrained column should show parse == 1.0 and halluc == 0.0 by
construction; the interesting numbers are whether tool_acc and args_exact rise
too (they do when masking removes structurally impossible tokens early).

  CUDA_VISIBLE_DEVICES=1 python scripts/tool_eval.py --run sft_tools_v3_363m --n 300
"""
import os, sys, glob, json, time, argparse
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import tool_data
import grammar as G
from train_gpu import build_model as _build
from train_sft import make_encoder_pair

DATA = os.path.join(ROOT, "training_data", "posttrain", "tools.jsonl")


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


@torch.no_grad()
def free_generate(model, prompt_ids, block_size, device, max_new, decode):
    """Unconstrained greedy decode, stopping at the EOS mark."""
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
        tok = int(model(x)[0, -1].argmax())
        out.append(tok); ids.append(tok)
        if tool_data.EOS_MARK in decode(out[-4:]):
            break
    return out


def score(pred_mode, pred_call, gold, tools):
    """Returns a dict of per-example 0/1 outcomes."""
    gold_mode = gold["target_mode"]
    offered = set(gold["tools"])
    r = {"parse": 0, "mode_acc": 0, "halluc": 0}
    r["parse"] = 1 if (pred_mode == tool_data.MODE_ANSWER or pred_call is not None) else 0
    r["mode_acc"] = 1 if pred_mode == gold_mode else 0
    if pred_call is not None and pred_call.get("name") not in offered:
        r["halluc"] = 1
    if gold_mode == tool_data.MODE_TOOL:
        gc = gold["target_call"]
        r["tool_acc"] = 1 if (pred_call and pred_call.get("name") == gc["name"]) else 0
        r["args_exact"] = 1 if (pred_call and r["tool_acc"]
                                and pred_call.get("arguments") == gc["arguments"]) else 0
    else:
        r["refuse_acc"] = 1 if pred_mode == tool_data.MODE_ANSWER else 0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=72)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modes", default="constrained,free")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    model = _build(c).to(args.device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    print(f"{args.run}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"step {blob.get('step')}, vocab {c['vocab']}")

    encode, decode = make_encoder_pair(c)
    tools = tool_data.load_tools()
    pairs, bad = tool_data.load_tool_jsonl(DATA, tools, split="test")
    pairs = pairs[:args.n]
    print(f"held-out test examples: {len(pairs)} (bad lines {bad})")
    if not pairs:
        print("no test data yet -- generate tools.jsonl first"); return

    table = G.TokenTable(c["vocab"])
    results = {}
    for mode in args.modes.split(","):
        agg, n_tool, n_ans, t0, shown = {}, 0, 0, time.time(), 0
        for p in pairs:
            gold = p["rec"]
            pid = encode(p["prompt"])
            if mode == "constrained":
                gr = G.ToolGrammar(gold["tools"], tools)
                out, text, _ = G.constrained_generate(
                    model, pid, gr, table, c["block_size"], args.device,
                    max_new=args.max_new, temperature=0.0)
            else:
                out = free_generate(model, pid, c["block_size"], args.device,
                                    args.max_new, decode)
                text = decode(out)
            pm, pc, _ = tool_data.parse_response(text)
            r = score(pm, pc, gold, tools)
            for k, v in r.items():
                agg[k] = agg.get(k, 0) + v
            n_tool += gold["target_mode"] == tool_data.MODE_TOOL
            n_ans += gold["target_mode"] == tool_data.MODE_ANSWER
            if shown < args.show:
                shown += 1
                print(f"  [{mode}] gold={gold['target_mode']} "
                      f"{gold.get('target_call', {}).get('name', '-')} | pred={text[:90]!r}")
        N = len(pairs)
        row = {
            "parse": agg.get("parse", 0) / N,
            "mode_acc": agg.get("mode_acc", 0) / N,
            "halluc": agg.get("halluc", 0) / N,
            "tool_acc": agg.get("tool_acc", 0) / max(1, n_tool),
            "args_exact": agg.get("args_exact", 0) / max(1, n_tool),
            "refuse_acc": agg.get("refuse_acc", 0) / max(1, n_ans),
            "sec": time.time() - t0,
        }
        results[mode] = row
        print(f"  {mode:12s} " + " ".join(f"{k}={v:.3f}" for k, v in row.items()
                                          if k != "sec")
              + f"  ({row['sec']:.0f}s)")

    print(f"\n{'metric':12s}" + "".join(f"{m:>14s}" for m in results))
    for k in ("parse", "mode_acc", "tool_acc", "args_exact", "refuse_acc", "halluc"):
        print(f"{k:12s}" + "".join(f"{results[m][k]:>14.3f}" for m in results))
    print(f"\n(n={len(pairs)} gold-TOOL + gold-ANSWER mixed; halluc lower is better)")


if __name__ == "__main__":
    main()
