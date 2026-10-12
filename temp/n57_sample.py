"""n57_sample.py <ckpt> -- 2026-10-12 (Neuro 57). Text samples from a pretraining checkpoint (is the language coherent?).
Sampling at temperature 0.7, top-k 40, 48 new tokens, fixed seed; plain continuation prompts (no chat format yet)."""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
from train_sft import make_encoder_pair
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False); c = dict(s["cfg"])
m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval(); del s
enc, dec = make_encoder_pair(dict(c, lut_subword_fallback=True))
torch.manual_seed(0)
PROMPTS = ["The capital of France is", "Once upon a time, a small dog", "To make a cup of tea, first",
           "The main reason plants need sunlight is", "In 1969, astronauts", "Question: What is 2 + 2?\nAnswer:"]
for p in PROMPTS:
    ids = enc(p)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(48):
            lg = m(torch.tensor([ids], device="cuda"))[0, -1].float() / 0.7
            v, _ = lg.topk(40); lg[lg < v[-1]] = float("-inf")
            ids.append(int(torch.multinomial(torch.softmax(lg, -1), 1)))
    print("SAMPLE |", dec(ids).replace("\n", " / "), flush=True)
