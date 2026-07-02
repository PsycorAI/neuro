"""Self-Refine demo: draft -> critique -> revise, N iterations.

Two modes:
  - persistent (default): Hebbian M-state is CARRIED across draft/critique/revise.
    The model 'remembers' its draft when critiquing. Our unique lever.
  - fresh: each step resets state (baseline transformer-like behavior for comparison).

Optional --brain loads a brain-mod state at start (e.g., writer.brain).
Post-Phase-1, we'll add a --critic-brain that swaps in for the critique step.

Test today on 350M_kd. This model is weak (val_ppl 35.5, vocab 16k with 10% OOV
on general text) so treat it as PIPELINE VALIDATION, not a quality flex. The
real self-refine value shows up when the 500M lands.

Usage:
  python scripts/self_refine.py --prompt "Explain photosynthesis." --iters 3
  python scripts/self_refine.py --prompt "..." --iters 3 --fresh
  python scripts/self_refine.py --prompt "..." --brain data/brains/python_350m_kd.brain
"""
import os, sys, yaml, argparse, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
from transformers import AutoTokenizer

CKPT = "/home/glenn/projects/neuro/models/checkpoints/phase3_350M_kd/step_0235882.pt"
CFG  = "/home/glenn/projects/neuro/configs/phase3_350M_kd.yaml"
LUT  = "/home/glenn/projects/neuro/data/vocab_map_16384.npy"
LLAMA = "meta-llama/Meta-Llama-3-8B"
PHI   = "microsoft/Phi-3.5-mini-instruct"

# When the model's vocab matches Phi's (32064) we skip the LUT projection
# entirely and use the Phi tokenizer directly — cleaner (no OOV, no inverse
# lookup dance). Used for the 500M and later.
PHI_VOCAB = 32064

GEN_LEN_DRAFT   = 120
GEN_LEN_CRITIQUE = 80
GEN_LEN_REVISE  = 120
TEMPERATURE = 0.6
TOP_K = 40
SEED_BASE = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_dev(state, device):
    if isinstance(state, dict): return {k: to_dev(v, device) for k, v in state.items()}
    if isinstance(state, list): return [to_dev(v, device) for v in state]
    if isinstance(state, torch.Tensor): return state.to(device)
    return state


def clone_state(state):
    """Deep-copy the Hebbian state dict so refine steps don't mutate the source."""
    if isinstance(state, dict): return {k: clone_state(v) for k, v in state.items()}
    if isinstance(state, list): return [clone_state(v) for v in state]
    if isinstance(state, torch.Tensor): return state.detach().clone()
    return state


class Encoder:
    """Handles Llama-3 -> student-vocab projection + inverse decode (16k mode),
    or direct Phi encode/decode (32k mode)."""
    def __init__(self, tok, lut=None):
        self.tok = tok
        self.lut = lut
        self.inv = None
        if lut is not None:
            inv = np.full(int(lut.max()) + 1, -1, dtype=np.int64)
            for li, si in enumerate(lut):
                if si > 0 and inv[si] == -1:
                    inv[si] = li
            self.inv = inv
    def encode(self, text):
        ids = np.asarray(self.tok.encode(text, add_special_tokens=False), dtype=np.int64)
        if self.lut is not None:
            ids = self.lut[ids]
        return torch.from_numpy(ids).unsqueeze(0)
    def decode(self, student_ids):
        arr = np.asarray(student_ids, dtype=np.int64)
        if self.inv is not None:
            arr = np.where(arr > 0, self.inv[arr], 0)
        return self.tok.decode(arr.tolist(), skip_special_tokens=True)


@torch.no_grad()
def generate(model, encoder, prompt_text, initial_state=None, gen_len=100,
             seed=SEED_BASE, persist_state=True):
    """Generate gen_len tokens after prompt_text.

    Returns (generated_text, final_state).
    If persist_state=True, final_state includes the Hebbian M evolved through
    both the prompt AND the generation. Otherwise state=None on return.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = encoder.encode(prompt_text).to(DEVICE)
    state = initial_state
    out_ids = []
    # Run the prompt through first so state is up to date
    logits, state = model(x, initial_state=state, return_final_state=True)
    # Sample the first output token from the last-prompt-position logits
    for step in range(gen_len):
        l = logits[0, -1] / TEMPERATURE
        if encoder.lut is not None:
            l[0] = float("-inf")   # mask UNK in LUT-projection mode only
        topv, topi = l.topk(TOP_K)
        probs = F.softmax(topv, dim=-1)
        idx = int(topi[torch.multinomial(probs, 1, generator=g).item()].item())
        out_ids.append(idx)
        # Feed the sampled token back to advance state + get next logits
        nx = torch.tensor([[idx]], device=DEVICE)
        logits, state = model(nx, initial_state=state, return_final_state=True)
    text = encoder.decode(out_ids)
    return text, (state if persist_state else None)


def self_refine(model, encoder, prompt, iters=3, persistent=True, brain=None,
                verbose=True):
    """Draft, then N rounds of critique -> revise. Optionally persistent state."""
    history = []
    state = brain  # start from brain-mod state if provided

    # --- 1. Draft ---
    draft_prompt = f"### Task:\n{prompt}\n\n### Response:\n"
    if verbose: print(f"\n=== Draft ===\n{draft_prompt}", flush=True)
    draft_text, state_after_draft = generate(
        model, encoder, draft_prompt, initial_state=state,
        gen_len=GEN_LEN_DRAFT, seed=SEED_BASE, persist_state=persistent)
    if verbose: print(f"[draft]: {draft_text!r}", flush=True)
    history.append(("draft", draft_text))

    current_text = draft_text
    for i in range(iters):
        # --- 2. Critique ---
        crit_state = state_after_draft if persistent else None
        crit_prompt = (
            f"### Task:\n{prompt}\n\n"
            f"### Response:\n{current_text}\n\n"
            f"### Critique (identify errors or improvements):\n"
        )
        if verbose: print(f"\n=== Iter {i+1} critique ===", flush=True)
        crit_text, state_after_crit = generate(
            model, encoder, crit_prompt, initial_state=clone_state(crit_state),
            gen_len=GEN_LEN_CRITIQUE, seed=SEED_BASE + i + 1,
            persist_state=persistent)
        if verbose: print(f"[critique {i+1}]: {crit_text!r}", flush=True)
        history.append((f"critique_{i+1}", crit_text))

        # --- 3. Revise ---
        rev_state = state_after_crit if persistent else None
        rev_prompt = (
            f"### Task:\n{prompt}\n\n"
            f"### Previous response:\n{current_text}\n\n"
            f"### Critique:\n{crit_text}\n\n"
            f"### Revised response:\n"
        )
        if verbose: print(f"\n=== Iter {i+1} revise ===", flush=True)
        rev_text, state_after_rev = generate(
            model, encoder, rev_prompt, initial_state=clone_state(rev_state),
            gen_len=GEN_LEN_REVISE, seed=SEED_BASE + 100 + i,
            persist_state=persistent)
        if verbose: print(f"[revision {i+1}]: {rev_text!r}", flush=True)
        history.append((f"revision_{i+1}", rev_text))

        current_text = rev_text
        state_after_draft = state_after_rev  # carry forward for next iter

    return current_text, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Task prompt.")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--fresh", action="store_true",
                    help="Reset Hebbian state each step (baseline mode).")
    ap.add_argument("--brain", default=None, help="Path to a .brain to load at start.")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--cfg", default=CFG)
    args = ap.parse_args()

    with open(args.cfg) as f:
        c = yaml.safe_load(f)
    print(f"loading {args.ckpt}  arch d={c['d']} n_neurons={c['n_neurons']} L={c['n_layers']}")
    model = build_model(c).to(DEVICE).eval()
    sd = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd["model"], strict=False)

    if c.get("vocab", 16384) == PHI_VOCAB:
        print("using Phi-3.5 tokenizer (no LUT)")
        tok = AutoTokenizer.from_pretrained(PHI)
        encoder = Encoder(tok, lut=None)
    else:
        print(f"using Llama-3 tokenizer + LUT (vocab={c.get('vocab')})")
        tok = AutoTokenizer.from_pretrained(LLAMA, token=os.environ.get("HF_TOKEN"))
        lut = np.load(LUT)
        encoder = Encoder(tok, lut)

    brain = None
    if args.brain:
        print(f"loading brain: {args.brain}")
        brain = to_dev(type(model).load_brain(args.brain), DEVICE)

    final, history = self_refine(model, encoder, args.prompt,
                                 iters=args.iters, persistent=(not args.fresh),
                                 brain=brain, verbose=True)

    print("\n" + "=" * 70)
    print(f"FINAL after {args.iters} iterations:")
    print(final)
    print("=" * 70)
    print(f"Mode: {'persistent Hebbian state' if not args.fresh else 'fresh state per step'}")


if __name__ == "__main__":
    main()
