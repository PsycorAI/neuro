"""n57_gen_profile.py -- 2026-10-12 (Neuro 57). Where does the fixed ~115 ms per generated token go? (Measured the same
at 189M and at 1.1B params and from 64 to 512 tokens of context, so it is overhead, not matmul time.)
cProfile of 20 batch-1 forwards with CUDA_LAUNCH_BLOCKING=1 (so GPU time lands on the calling Python function).
  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python temp/n57_gen_profile.py sft_v4_reh13 128"""
import sys, cProfile, pstats, io, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
torch.set_float32_matmul_precision("high")
name, T = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{name}.yaml"))
m = build_model(c).to("cuda").eval()
x = torch.randint(1, c["vocab"], (1, T), device="cuda")
with torch.no_grad():
    for _ in range(3):
        m(x)
    pr = cProfile.Profile(); pr.enable()
    for _ in range(20):
        m(x)
    torch.cuda.synchronize(); pr.disable()
for key in ("cumulative", "tottime"):
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(key).print_stats(28)
    print(f"===== sorted by {key} (20 forwards, context {T}) =====")
    print("\n".join(line[:170] for line in s.getvalue().splitlines()[4:]))
