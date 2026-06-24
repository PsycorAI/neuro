"""Interactive chat with an SFT'd PsycorNeuro checkpoint.

Uses the same Alpaca chat template as src/sft_data.py:
    ### Instruction:
    {message}

    ### Response:
    ...

Stops generation on "###" (the response terminator we trained with).

  python scripts/chat.py --run sft_30m_mds                       # interactive REPL
  python scripts/chat.py --run sft_30m_mds --message "Hi"        # one-shot
  python scripts/chat.py --run sft_30m_mds --temperature 0.8 --max_tokens 200
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
LUT_PATH = os.path.join(ROOT, "data", "vocab_map_16384.npy")

PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"
STOP_MARKER = "###"


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"], d_mem=c["d_mem"],
                            recurrent=c.get("recurrent", False),
                            rec_density=c.get("rec_density", 0.05),
                            compile_safe=c.get("compile", False),
                            tie_weights=c.get("tie_weights", False),
                            n_layers=c.get("n_layers", 1),
                            use_fpt=c.get("use_fpt", False),
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
    inv = np.full(16384, -1, dtype=np.int64)
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return [int(lut[t]) for t in ids if lut[t] != 0]
    def decode(ids):
        return tk.decode([int(inv[i]) for i in ids if 0 < i < 16384 and inv[i] >= 0])
    return encode, decode


@torch.no_grad()
def generate_response(model, c, message, encode, decode, device,
                      max_tokens, temperature, topk, repetition_penalty):
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
        logits = model(x)[0, -1].float()
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
    args = ap.parse_args()

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    m = build(c, args.device); m.load_state_dict(blob["model"], strict=False); m.eval()
    print(f"loaded {args.run}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, "
          f"{blob.get('tokens',0)/1e6:.0f}M tok\n")
    encode, decode = load_tokenizer()

    def chat_once(msg):
        return generate_response(m, c, msg, encode, decode, args.device,
                                  args.max_tokens, args.temperature, args.topk,
                                  args.repetition_penalty)

    if args.message is not None:
        print(f"USER: {args.message}\nASSISTANT: {chat_once(args.message)}")
        return

    print("Chat mode. Empty line to quit.\n")
    while True:
        try: msg = input("USER: ").strip()
        except EOFError: break
        if not msg: break
        print(f"ASSISTANT: {chat_once(msg)}\n")


if __name__ == "__main__":
    main()
