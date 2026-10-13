"""brain_code_quiz.py -- 2026-10-12 (Neuro 58; Python brain = a skill). Run-the-code quiz for a coding "brain".
Each quiz request (training_data/brains/python/quiz.jsonl, scripts/gen_pybrain_data.py) asks for one named function;
the model answers in ANSWER mode (forced by the grammar); the reply is executed with the request's tests in a
separate Python process (-I, 5 s limit). pass = the tests run without error. Reported per set (taught = trained task
families in unseen wordings, control = families never trained) for the base model and with each adapter merged.
  python scripts/brain_code_quiz.py --base sft_v4_reh13_v2tr --quiz training_data/brains/python/quiz.jsonl \
         --brain py=brain_py13 [--n 90]"""
import os, sys, glob, json, argparse, random, subprocess
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
ap.add_argument("--n", type=int, default=90, help="requests per quiz set")
ap.add_argument("--max-new", type=int, default=130)
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


def runs_ok(code, tests):
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code + "\n" + tests], capture_output=True, timeout=5, cwd="/tmp")
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False


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
            code = text.split("###")[0].strip("\n")
            if code.startswith(" ") and not code.startswith("  "): code = code[1:]      # the space after "ANSWER"
            ok = runs_ok(code, r["tests"]); hit += ok
            if i < 2:
                print(f"    [{label}/{k}] {r['instruction'][:70]!r} -> {'PASS' if ok else 'FAIL'} {code[:110]!r}", flush=True)
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
