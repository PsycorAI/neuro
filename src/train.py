"""Phase-1 smoke trainer on the induction task: prove the spiking Hebbian core
learns, that the fast-weight memory is NECESSARY (ablation -> chance), and that
spikes stay sparse (most neuron-timesteps are silent)."""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikingHebbianLM
from data import RepeatTask


def evaluate(model, task, B, device, ablate):
    model.eval()
    with torch.no_grad():
        x, y, second = task.batch(B, device)
        logits, sr = model(x, ablate_memory=ablate, return_stats=True)
        pred = logits.argmax(-1)
        acc = (pred[:, second] == y[:, second]).float().mean().item()
    return acc, sr.item()


def main():
    torch.manual_seed(0)
    device = "cpu"   # Phase-1 smoke on CPU: live BDH training on the 5080 stays untouched
    task = RepeatTask(seq_len=8, n_symbols=20)
    model = SpikingHebbianLM(task.vocab, d=128, n_neurons=256, d_mem=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    steps, B = 1500, 128
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y, _ = task.batch(B, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            acc, srv = evaluate(model, task, 512, device, ablate=False)
            print(f"step {step:4d} | loss {loss.item():.3f} | recall_acc {acc:.3f} | spike_rate {srv:.3f}")

    acc_mem, sr = evaluate(model, task, 2000, device, ablate=False)
    acc_abl, _ = evaluate(model, task, 2000, device, ablate=True)
    chance = 1.0 / task.S
    print("\n=== Phase-1 smoke gates (induction/repeat task) ===")
    print(f"params {n_params:,} | {time.time()-t0:.1f}s on {device.upper()}")
    print(f"G3 memory-is-real : acc_with_M={acc_mem:.3f}  acc_ablated={acc_abl:.3f}  chance={chance:.3f}")
    sparsity = 1.0 - sr
    print(f"G2 sparsity       : {sparsity:.1%} silent (spike_rate={sr:.3f}; target >=70%, cf. SpikingBrain 69%)")
    print(f"G5 consumer-HW    : ran on {device.upper()} (no CUDA)")
    g3 = acc_mem > 0.8 and acc_abl < 3 * chance
    g2 = sparsity >= 0.70
    print(f"RESULT: G3={'PASS' if g3 else 'FAIL'}  G2={'PASS' if g2 else 'FAIL'}")


if __name__ == "__main__":
    main()
