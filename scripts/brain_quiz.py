"""brain_quiz.py -- 2026-10-12 (Neuro 57, docs/demo_v4_plan_2026_10_12.md D7). Closed-book quiz for a "brain" (a LoRA
adapter trained from documents). For each quiz question the model answers in ANSWER mode (forced by the grammar, no
search tool offered); correct = the gold string appears in the reply. Reported per quiz set (taught / control / ...)
for the base model alone and with each adapter merged into the weights.
  python scripts/brain_quiz.py --base sft_v4_reh13 --quiz training_data/brains/mech/quiz.jsonl \
         --brain mech=brain_mech13 [--brain other=run] [--n 150]"""
import os, sys, glob, json, argparse, random
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import tool_data, grammar as G, lora
from train_gpu import build_model
from train_sft import make_encoder_pair

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--quiz", required=True)
ap.add_argument("--brain", action="append", default=[], help="label=run (uses the newest adapter_step_*.pt of the run)")
ap.add_argument("--n", type=int, default=150, help="questions per quiz set")
ap.add_argument("--max-new", type=int, default=24)
a = ap.parse_args()


def latest(run, pat="step_*.pt"):
    return sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, pat)))[-1]


blob = torch.load(latest(a.base), map_location="cuda", weights_only=False)
c = blob["cfg"]
encode, decode = make_encoder_pair(c)
tools = tool_data.load_tools()
table = G.TokenTable(c["vocab"])
sets = {}
for line in open(os.path.join(ROOT, a.quiz) if not os.path.isabs(a.quiz) else a.quiz, encoding="utf-8"):
    r = json.loads(line); sets.setdefault(r["set"], []).append(r)
for k in sets:
    random.Random(0).shuffle(sets[k]); sets[k] = sets[k][:a.n]


def fresh():
    m = build_model(c).to("cuda")
    m.load_state_dict(blob["model"], strict=False)
    return m.eval()


@torch.no_grad()
def quiz(m, label):
    out = {}
    for k, rows in sets.items():
        hit = 0
        for i, r in enumerate(rows):
            gr = G.ToolGrammar(r["tools"], tools, proto=c.get("tool_proto", 1))
            pid = encode(tool_data.format_prompt(r["instruction"], r["tools"], tools) + tool_data.MODE_ANSWER)
            gr.root = gr.walk(gr.root, tool_data.MODE_ANSWER)          # the answer mode is forced
            _, text, _ = G.constrained_generate(m, pid, gr, table, c["block_size"], "cuda", max_new=a.max_new)
            ok = r["gold"].lower() in text.lower(); hit += ok
            if i < 2:
                print(f"    [{label}/{k}] {r['instruction'][:60]!r} gold={r['gold']!r} -> {text.strip()[:60]!r}", flush=True)
        out[k] = hit / max(1, len(rows))
    print(f"  quiz {label:12s} " + " | ".join(f"{k} {v * 100:5.1f}% (n={len(sets[k])})" for k, v in out.items()), flush=True)
    return out


quiz(fresh(), "base")
for spec in a.brain:
    label, run = spec.split("=", 1)
    ad = torch.load(latest(run, "adapter_step_*.pt"), map_location="cuda", weights_only=False)
    m = fresh()
    lora.inject_lora(m, r=ad["lora_r"], alpha=ad["lora_alpha"], targets=tuple(ad["lora_targets"]))
    m.to("cuda")
    lora.load_lora_state_dict(m, ad["lora"])
    n = lora.merge_lora(m)
    print(f"  brain {label}: {sum(v.numel() for v in ad['lora'].values()) / 1e6:.2f}M adapter params, merged into {n} layers")
    quiz(m.eval(), label)
