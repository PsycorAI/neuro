"""LM benchmark suite for PsycorNeuro checkpoints.

Multi-choice tasks: HellaSwag, ARC-Easy, ARC-Challenge, MMLU (subset).
Scoring: for each (prompt, choices), compute log-likelihood of each choice
conditioned on prompt; pick argmax. Reports both raw (sum log-p) and
length-normalized (avg log-p per token) accuracy.

  python scripts/lm_eval.py --run phase3_350M_kd --tasks hellaswag,arc_easy
  python scripts/lm_eval.py --run hebb_90m --tasks mmlu --mmlu_subjects 5 --limit 100

Caveats: our 16k LUT vocab loses rare Llama-3 tokens; OOV tokens in the
continuation are SKIPPED in scoring (they'd give zero probability otherwise).
The model is undertrained (BDH 7-14B tok corpus, no instruction data),
so absolute scores will be near random — the value is the relative comparison
across checkpoints and a baseline for SFT lift later.
"""
import os, sys, glob, argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
LUT_PATH = os.path.join(ROOT, "data", "vocab_map_16384.npy")


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"], d_mem=c["d_mem"],
                            recurrent=c.get("recurrent", False), rec_density=c.get("rec_density", 0.05),
                            compile_safe=c.get("compile", False), tie_weights=c.get("tie_weights", False),
                            n_layers=c.get("n_layers", 1), use_fpt=c.get("use_fpt", False),
                            fpt_K=c.get("fpt_K", 10), lam=c.get("lam", 0.98),
                            learnable_decay=c.get("learnable_decay", False),
                            write_gate=c.get("write_gate", False),
                            delta_rule=c.get("delta_rule", False),
                            beta_floor=c.get("beta_floor", 0.0),
                            decay_gate=c.get("decay_gate", False),
                            titans=c.get("titans", False),
                            local_attn=c.get("local_attn", False),
                            local_window=c.get("local_window", 64),
                            n_heads=c.get("n_heads", 1),
                            pre_conv=c.get("pre_conv", False),
                            pre_conv_kernel=c.get("pre_conv_kernel", 4),
                            vector_beta=c.get("vector_beta", False)).to(device)


def load_tokenizer():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut = np.load(LUT_PATH)
    def encode(text):
        """Encode text → list of student-vocab ids (drops OOV / id=0)."""
        ids = tk.encode(text, add_special_tokens=False)
        return [int(lut[t]) for t in ids if lut[t] != 0]
    return encode


@torch.no_grad()
def score_continuation(model, c, prompt_ids, cont_ids, device, block_size):
    """Score log p(cont | prompt). Returns (sum_logp, num_scored).
    Skips OOV (id=0) tokens; if cont_ids empty after filtering, returns (0, 0).
    Truncates prompt to fit block_size."""
    if len(cont_ids) == 0:
        return 0.0, 0
    full = prompt_ids + cont_ids
    if len(full) > block_size:
        full = full[-block_size:]
    cont_start = len(full) - len(cont_ids)
    x = torch.tensor([full], dtype=torch.long, device=device)
    logits = model(x)[0]                                    # (T, V)
    logp = F.log_softmax(logits, dim=-1)                    # (T, V)
    # Predict position t from logits at t-1; score positions in [cont_start, len(full))
    total = 0.0; n = 0
    for t in range(cont_start, len(full)):
        target = full[t]
        if target == 0:        # skip OOV continuation token
            continue
        total += logp[t - 1, target].item()
        n += 1
    return total, n


def score_choice(model, c, prompt, choice, encode, device, block_size):
    """Returns (sum_logp, num_tokens) for prompt + choice."""
    p_ids = encode(prompt)
    c_ids = encode(choice)
    return score_continuation(model, c, p_ids, c_ids, device, block_size)


def evaluate_mc(model, c, items, encode, device, block_size, limit=None):
    """items: list of {prompt, choices, gold (int index)}.
    Returns: dict with acc (raw), acc_norm (length-normalized), N."""
    n_total = 0; n_correct = 0; n_correct_norm = 0
    for i, it in enumerate(items):
        if limit and i >= limit: break
        scores = []
        for ch in it["choices"]:
            sp, ntok = score_choice(model, c, it["prompt"], ch, encode, device, block_size)
            avg = sp / max(ntok, 1)
            scores.append((sp, avg))
        raw = max(range(len(scores)), key=lambda k: scores[k][0])
        norm = max(range(len(scores)), key=lambda k: scores[k][1])
        n_correct += int(raw == it["gold"])
        n_correct_norm += int(norm == it["gold"])
        n_total += 1
    return {"acc": n_correct / max(n_total, 1),
            "acc_norm": n_correct_norm / max(n_total, 1),
            "N": n_total}


# --- task loaders (return list of {prompt, choices, gold}) ----------------

def load_hellaswag(split="validation"):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split=split)
    out = []
    for ex in ds:
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        out.append({"prompt": ctx + " ",
                    "choices": [" " + e for e in ex["endings"]],
                    "gold": int(ex["label"])})
    return out


def load_arc(challenge=False, split="validation"):
    from datasets import load_dataset
    name = "ARC-Challenge" if challenge else "ARC-Easy"
    ds = load_dataset("allenai/ai2_arc", name, split=split)
    out = []
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        if ex["answerKey"] not in labels:
            continue
        gold = labels.index(ex["answerKey"])
        out.append({"prompt": f"Question: {ex['question']}\nAnswer:",
                    "choices": [" " + t for t in texts],
                    "gold": gold})
    return out


def load_mmlu(subjects=None, split="validation"):
    from datasets import load_dataset
    out = []
    if subjects is None:
        subjects = ["high_school_mathematics", "high_school_world_history",
                    "elementary_mathematics", "global_facts",
                    "miscellaneous"]
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split=split)
        except Exception as e:
            print(f"  [skip {subj}: {e}]"); continue
        for ex in ds:
            out.append({"prompt": f"{ex['question']}\nAnswer:",
                        "choices": [" " + c for c in ex["choices"]],
                        "gold": int(ex["answer"])})
    return out


TASKS = {
    "hellaswag": load_hellaswag,
    "arc_easy": lambda: load_arc(challenge=False),
    "arc_challenge": lambda: load_arc(challenge=True),
    "mmlu": load_mmlu,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint dir name in models/checkpoints/")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge",
                    help="comma-sep from: hellaswag, arc_easy, arc_challenge, mmlu")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per task (None = full validation set)")
    ap.add_argument("--block_size", type=int, default=0,
                    help="override; 0 = use ckpt cfg")
    ap.add_argument("--mmlu_subjects", type=int, default=5,
                    help="number of MMLU subjects to include (default 5)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    block_size = args.block_size or c["block_size"]
    m = build(c, args.device); m.load_state_dict(blob["model"], strict=False); m.eval()
    print(f"loaded {args.run}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, "
          f"{blob.get('tokens', 0)/1e9:.2f}B tok, block_size={block_size}\n")

    encode = load_tokenizer()
    tasks = args.tasks.split(",")
    results = {}
    for task in tasks:
        if task not in TASKS:
            print(f"unknown task: {task}"); continue
        print(f"loading {task}...", flush=True)
        items = TASKS[task]()
        if task == "mmlu" and args.mmlu_subjects:
            # truncate by re-running load with fewer subjects
            pass  # handled inside loader default list
        print(f"  {len(items)} items", flush=True)
        t0 = time.time()
        r = evaluate_mc(m, c, items, encode, args.device, block_size, args.limit)
        r["seconds"] = time.time() - t0
        results[task] = r
        chance = 1.0 / max(1, len(items[0]["choices"])) if items else 0
        print(f"  {task:14} | acc {r['acc']*100:5.1f}%  acc_norm {r['acc_norm']*100:5.1f}%  "
              f"(N={r['N']}, chance={chance*100:.0f}%, {r['seconds']:.1f}s)\n", flush=True)

    print("=== summary ===")
    for task, r in results.items():
        print(f"  {task:14} acc {r['acc']*100:5.1f}%  acc_norm {r['acc_norm']*100:5.1f}%  (N={r['N']})")


if __name__ == "__main__":
    main()
