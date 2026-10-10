# Tiny CUDA health check (10-08): allocate, matmul, free, synchronize on the visible GPU.
import torch
x = torch.randn(4096, 4096, device="cuda")
for _ in range(20):
    y = x @ x
z = [torch.empty(256 * 1024 * 1024 // 4, device="cuda") for _ in range(8)]   # 8 x 256 MB
del z
torch.cuda.synchronize()
print("[OK]", torch.cuda.get_device_name(), "alloc conf:", __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF", "-"),
      f"max reserved {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")
