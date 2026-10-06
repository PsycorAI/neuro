# diag_tfmr_mqar.py -- 2026-09-26: why does the 30M fixed transformer (screen_tfmrfix_s0_30m)
# sit at chance on MQAR (train MQAR CE ~9 = ln V) while LM ppl trains to 88?
# Train the SAME 2L all-attention model on MQAR ONLY (N in [4,16]) for 600 steps, B=32:
#   arm "adamw" : plain AdamW lr 1e-3               -> is the block/model able to learn MQAR?
#   arm "recipe": train_gpu._build_optimizers(cfg)  -> does the project optimizer recipe break it?
# Pass = MQAR CE falls well below ln(16384)=9.7 (a working attention stack solves N<=16 fast).
import sys, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "src")
import train_gpu as T
torch.manual_seed(0)
c = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "configs/screen_tfmrfix_s0_30m.yaml"))
arm = sys.argv[1]
m = T.build_model(c).cuda().train()
if arm == "adamw":
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)]
else:
    opts = T._build_optimizers(m, c, "cuda")
for step in range(1, 601):
    N = int(torch.randint(4, 17, (1,)))
    x, y = T.mqar_aug_batch(32, N, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce = F.cross_entropy(m(x)[:, -1, :].float(), y)
    for o in opts: o.zero_grad(set_to_none=True)
    ce.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    for o in opts: o.step()
    if step % 100 == 0:
        print(f"[{arm}] step {step} N={N} mqar_ce {ce.item():.3f}", flush=True)
