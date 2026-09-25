"""Interactive chat with an SFT'd PsycorNeuro checkpoint.

Uses the same Alpaca chat template as src/sft_data.py:
    ### Instruction:
    {message}

    ### Response:
    ...

Stops generation on "###" (the response terminator we trained with).

Persistent Hebbian brain state carries across turns within a REPL session
(the project's unique lever -- Hebbian memory M accumulates as you talk).
Optional pre-baked brain loading via --load_brain path/to/knowledge.brain.
REPL commands: /reset-brain, /save-brain PATH, /load-brain PATH.

  python scripts/chat.py --run sft_30m_mds                       # interactive REPL
  python scripts/chat.py --run sft_30m_mds --message "Hi"        # one-shot
  python scripts/chat.py --run sft_30m_mds --temperature 0.8 --max_tokens 200
  python scripts/chat.py --run sft_500m_tulu3 --load_brain models/brains/sft_500m_tulu3/knowledge.brain
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_gpu import build_model   # canonical builder — do not re-derive here
from model import SpikingHebbianLM   # for load_brain static method

ROOT = os.path.join(os.path.dirname(__file__), "..")

PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"
STOP_MARKER = "###"


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    # Delegate to train_gpu.build_model so this stays in sync as new arch
    # flags (engram_v2, mtp, gdn2, ...) are added. Previously chat.py had its
    # own stale copy that silently dropped ~163M params on 500M ckpts.
    return build_model(c).to(device)


def load_tokenizer(vocab):
    # 2026-08-07: parametrized on ckpt vocab (was hardcoded 16k) and
    # PRESERVES id 0 in encoded sequences instead of dropping — matches
    # training-time data_stream.py behavior. See docs/polish_recommendations_2026_08_07.md.
    from transformers import AutoTokenizer
    tk_name = "microsoft/Phi-3.5-mini-instruct" if vocab == 32064 else "NousResearch/Meta-Llama-3-8B"
    lut_path = os.path.join(ROOT, "data", f"vocab_map_{vocab}.npy")
    tk = AutoTokenizer.from_pretrained(tk_name)
    lut = np.load(lut_path)
    inv = np.full(vocab, -1, dtype=np.int64)
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return [int(lut[t]) for t in ids]   # keep id 0 — matches training
    def decode(ids):
        parts = []
        for i in ids:
            i = int(i)
            if 0 < i < vocab and inv[i] >= 0:
                parts.append(tk.decode([int(inv[i])]))
            else:
                parts.append("<unk>")   # visible marker for id 0 / OOV
        return "".join(parts)
    return encode, decode


@torch.no_grad()
def generate_response(model, c, message, encode, decode, device,
                      max_tokens, temperature, topk, repetition_penalty,
                      initial_state=None):
    prompt = PROMPT_FMT.format(message=message)
    p_ids = encode(prompt)
    if not p_ids:
        return "[prompt was all-OOV]"
    out_ids = list(p_ids)
    generated = []
    stop_ids = encode(STOP_MARKER)        # detect "###" to stop
    block_size = c["block_size"]

    for _ in range(max_tokens):
        x = torch.tensor([out_ids[-block_size:]], dtype=torch.long, device=device)
        out = model(x, initial_state=initial_state)
        logits = out[0, -1].float() if not isinstance(out, tuple) else out[0][0, -1].float()
        if repetition_penalty > 1.0 and generated:
            for tok in set(generated[-64:]):
                logits[tok] /= repetition_penalty
        logits = logits / max(temperature, 1e-6)
        if topk:
            v, _ = torch.topk(logits, topk)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        out_ids.append(nxt)
        generated.append(nxt)
        # Stop if recent tokens spell the STOP_MARKER
        if len(generated) >= len(stop_ids):
            tail = generated[-len(stop_ids):]
            if tail == stop_ids:
                generated = generated[:-len(stop_ids)]
                break
    return decode(generated).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SFT'd checkpoint dir name")
    ap.add_argument("--message", default=None, help="one-shot message; omit for REPL")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--load_brain", default=None,
                    help="path to a pre-baked .brain file to seed persistent state")
    ap.add_argument("--no_persist", action="store_true",
                    help="disable persistent Hebbian state carry-over between turns")
    args = ap.parse_args()

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    m = build(c, args.device); m.load_state_dict(blob["model"], strict=False); m.eval()
    print(f"loaded {args.run}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, "
          f"{blob.get('tokens',0)/1e6:.0f}M tok")
    encode, decode = load_tokenizer(c["vocab"])

    # Persistent Hebbian brain state, carried across turns.
    persistent = {"state": None}
    if args.load_brain:
        persistent["state"] = SpikingHebbianLM.load_brain(args.load_brain)
        print(f"loaded brain: {args.load_brain}")
    if args.no_persist:
        print("persistent brain: DISABLED (--no_persist)")
    print()

    @torch.no_grad()
    def chat_once(msg):
        state_in = None if args.no_persist else persistent["state"]
        resp = generate_response(m, c, msg, encode, decode, args.device,
                                  args.max_tokens, args.temperature, args.topk,
                                  args.repetition_penalty, initial_state=state_in)
        if not args.no_persist:
            # Commit pass: update persistent state with this turn's full content.
            # The generation loop only READS state; we advance it here so the
            # next turn sees an updated brain.
            full_txt = PROMPT_FMT.format(message=msg) + resp
            full_ids = encode(full_txt)
            if full_ids:
                x = torch.tensor([full_ids[-c["block_size"]:]], dtype=torch.long,
                                 device=args.device)
                out = m(x, initial_state=persistent["state"], return_final_state=True)
                # model.forward returns (logits, state) when return_final_state=True
                persistent["state"] = out[1] if isinstance(out, tuple) else None
        return resp

    if args.message is not None:
        print(f"USER: {args.message}\nASSISTANT: {chat_once(args.message)}")
        return

    print("Chat mode. Empty line to quit.")
    print("Commands: /reset-brain, /save-brain PATH, /load-brain PATH\n")
    while True:
        try: msg = input("USER: ").strip()
        except EOFError: break
        if not msg: break
        if msg == "/reset-brain":
            persistent["state"] = None
            print("[brain reset]\n"); continue
        if msg.startswith("/save-brain "):
            path = msg.split(" ", 1)[1].strip()
            if persistent["state"] is None:
                print("[no brain state to save]\n"); continue
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            m.save_brain(persistent["state"], path)
            print(f"[saved brain to {path}]\n"); continue
        if msg.startswith("/load-brain "):
            path = msg.split(" ", 1)[1].strip()
            persistent["state"] = SpikingHebbianLM.load_brain(path)
            print(f"[loaded brain from {path}]\n"); continue
        print(f"ASSISTANT: {chat_once(msg)}\n")


if __name__ == "__main__":
    main()
