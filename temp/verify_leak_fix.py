import sys, subprocess, importlib.util, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_parallel as NEW
src = subprocess.run(["git", "-C", "/home/glenn/projects/neuro", "show", "HEAD:src/lif_parallel.py"], capture_output=True, text=True, check=True).stdout
open("/tmp/lif_old.py", "w").write(src)
sp = importlib.util.spec_from_file_location("lif_old", "/tmp/lif_old.py"); OLD = importlib.util.module_from_spec(sp); sp.loader.exec_module(OLD)
torch.manual_seed(0)
for dt in (torch.float32, torch.bfloat16):
    x = (torch.randn(2, 512, 128, device="cuda") * 0.5).to(dt)
    yn, yo = NEW._scan_chunked(0.9, x), OLD._scan_chunked(0.9, x)
    leak = 0.0
    for p in (258, 260, 300, 400, 510):
        xp = x.clone(); xp[:, p] += 5.0
        leak = max(leak, (NEW._scan_chunked(0.9, xp).float() - yn.float())[:, :p].abs().max().item())
    print(f"{dt}: new==old overall {torch.equal(yn, yo)} | new==old for t<256 {torch.equal(yn[:, :256], yo[:, :256])} | max future leak (new) {leak:.3g}")
    cur = (torch.randn(2, 512, 128, device="cuda") * 0.6 + 0.1).to(dt)
    sn, so = NEW.lif_parallel(cur, 0.9, K=10)[0], OLD.lif_parallel(cur, 0.9, K=10)[0]
    print(f"   lif_parallel spikes t<256 identical: {torch.equal(sn[:, :256], so[:, :256])}")
