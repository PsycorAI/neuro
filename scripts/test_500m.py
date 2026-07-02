"""Test the 500M with prompts. Uses Phi-3.5 native tokenizer, no LUT.

Auto-detects the latest checkpoint in models/checkpoints/demo_phase1_500m/.

Usage:
  python scripts/test_500m.py --prompt "Explain photosynthesis."
  python scripts/test_500m.py --prompt "def fibonacci(n):"
  python scripts/test_500m.py                                     # interactive REPL
  python scripts/test_500m.py --brain data/brains/python_500m.brain --prompt "def"
  python scripts/test_500m.py --ckpt models/checkpoints/demo_phase1_500m/step_0124000.pt

Options:
  --temp 0.6 --top_k 40 --gen_len 120 --seed 0
"""
import os, sys, glob, yaml, argparse, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
from transformers import AutoTokenizer

CKPT_DIR = "/home/glenn/projects/neuro/models/checkpoints/demo_phase1_500m"
CFG_PATH = "/home/glenn/projects/neuro/configs/demo_phase1_500m.yaml"
PHI_ID   = "microsoft/Phi-3.5-mini-instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def latest_ckpt(d):
    ckpts = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {d}")
    return ckpts[-1]


def to_dev(state, device):
    if isinstance(state, dict): return {k: to_dev(v, device) for k, v in state.items()}
    if isinstance(state, list): return [to_dev(v, device) for v in state]
    if isinstance(state, torch.Tensor): return state.to(device)
    return state


@torch.no_grad()
def generate(model, tok, prompt, brain=None, gen_len=120, temp=0.6, top_k=40, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], device=DEVICE)
    state = brain
    out_ids = []
    logits, state = model(x, initial_state=state, return_final_state=True)
    for _ in range(gen_len):
        l = logits[0, -1] / temp
        topv, topi = l.topk(top_k)
        probs = F.softmax(topv, dim=-1)
        idx = int(topi[torch.multinomial(probs, 1, generator=g).item()].item())
        out_ids.append(idx)
        nx = torch.tensor([[idx]], device=DEVICE)
        logits, state = model(nx, initial_state=state, return_final_state=True)
    return tok.decode(out_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help="Single prompt (else REPL).")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path (else latest).")
    ap.add_argument("--cfg", default=CFG_PATH)
    ap.add_argument("--brain", default=None, help="Path to a .brain to load.")
    ap.add_argument("--gen_len", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.cfg) as f:
        c = yaml.safe_load(f)
    print(f"arch: d={c['d']} n_neurons={c['n_neurons']} L={c['n_layers']} vocab={c['vocab']}")
    model = build_model(c).to(DEVICE).eval()

    ckpt = args.ckpt or latest_ckpt(CKPT_DIR)
    print(f"loading: {ckpt}")
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    step = sd.get("step", "?"); tokens = sd.get("tokens", "?")
    model.load_state_dict(sd["model"], strict=False)
    print(f"  step={step}  tokens={tokens/1e9:.2f}B" if isinstance(tokens, int)
          else f"  step={step}  tokens={tokens}")

    print("loading Phi-3.5 tokenizer...")
    tok = AutoTokenizer.from_pretrained(PHI_ID)

    brain = None
    if args.brain:
        print(f"loading brain: {args.brain}")
        brain = to_dev(type(model).load_brain(args.brain), DEVICE)

    def run(prompt, seed):
        text = generate(model, tok, prompt, brain=brain, gen_len=args.gen_len,
                        temp=args.temp, top_k=args.top_k, seed=seed)
        print("-" * 70)
        print(f"PROMPT: {prompt!r}")
        print("-" * 70)
        print(f"OUTPUT: {text}")
        print("-" * 70)

    if args.prompt is not None:
        run(args.prompt, args.seed)
    else:
        print("\nInteractive REPL. Empty line or Ctrl-D to quit.")
        i = 0
        while True:
            try:
                p = input("\n> ").strip()
            except EOFError:
                break
            if not p: break
            run(p, args.seed + i)
            i += 1


if __name__ == "__main__":
    main()
