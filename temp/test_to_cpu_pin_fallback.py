# test_to_cpu_pin_fallback.py -- 2026-09-26: _to_cpu falls back to a pageable copy when the
# pinned alloc fails (simulated), and still pins normally otherwise. Uses a tiny CUDA tensor.
import sys, torch
sys.path.insert(0, "src")
import train_gpu as T
x = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
real = torch.empty_like
def boom(*a, **k):
    if k.get("pin_memory"):
        raise torch.AcceleratorError("CUDA error: out of memory") if hasattr(torch, "AcceleratorError") else RuntimeError("oom")
    return real(*a, **k)
torch.empty_like = boom
y = T._to_cpu({"w": x}, ("t",))["w"]
torch.empty_like = real
assert y.device.type == "cpu" and not y.is_pinned() and torch.equal(y, x.cpu()), "fallback wrong"
assert ("t", "w") not in T._pinned_cache, "failed alloc must not be cached"
z = T._to_cpu({"w": x}, ("t2",))["w"]
assert z.is_pinned() and torch.equal(z, x.cpu()), "normal pinned path broken"
print("PASS", issubclass(getattr(torch, "AcceleratorError", RuntimeError), RuntimeError))
