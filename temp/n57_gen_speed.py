"""n57_gen_speed.py -- 2026-10-12 (Neuro 57). Generation speed of the demo path as it is today: one FULL forward per new
token, batch 1, no KV cache (step_kv does not support attn_canon or more than one attention chunk, hybrid_attn.py:491).
Random-init model built from a config (speed does not depend on the weights), so no checkpoint is needed.
Reports ms per token at several context lengths, in fp32 (what chat.py / demo/server.py do today) and bf16 autocast.
  CUDA_VISIBLE_DEVICES=0 python temp/n57_gen_speed.py v4_cloud_conv sft_v4_reh13"""
import sys, time, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
torch.set_float32_matmul_precision("high")
for name in sys.argv[1:] or ["v4_cloud_conv"]:
    c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{name}.yaml"))
    m = build_model(c).to("cuda").eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{name}: {n:.0f}M params on {torch.cuda.get_device_name(0)}, {torch.cuda.memory_allocated() / 2**30:.1f} GB",
          flush=True)
    for amp in (False, True):
        for T in (64, 128, 256, 512, 1024, 1536, 2048):
            if T > c["block_size"]:
                continue
            x = torch.randint(1, c["vocab"], (1, T), device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in range(3):
                    m(x)
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(10):
                    m(x)
                torch.cuda.synchronize()
            ms = (time.time() - t0) / 10 * 1000
            print(f"  {'bf16' if amp else 'fp32'} context {T:5d}: {ms:7.1f} ms per token = {1000 / ms:5.1f} tok/s  "
                  f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
    del m
    torch.cuda.empty_cache()
