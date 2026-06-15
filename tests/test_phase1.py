"""Phase-1 gates as tests. Run: pytest -q (uses the bdh venv)."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from data import RepeatTask


def _train_recall(steps=500):
    torch.manual_seed(0)
    task = RepeatTask(seq_len=8, n_symbols=20)
    m = SpikingHebbianLM(task.vocab)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(steps):
        x, y, _ = task.batch(128)
        loss = F.cross_entropy(m(x).reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m, task


def _recall_acc(m, task, ablate):
    x, y, second = task.batch(512)
    with torch.no_grad():
        pred = m(x, ablate_memory=ablate).argmax(-1)
    return (pred[:, second] == y[:, second]).float().mean().item()


def test_memory_is_necessary():
    m, task = _train_recall()
    with_mem = _recall_acc(m, task, ablate=False)
    ablated = _recall_acc(m, task, ablate=True)
    assert with_mem > 0.8, f"learned recall too low: {with_mem}"
    assert ablated < 0.15, f"ablation should collapse to ~chance(0.05): {ablated}"


def test_runs_on_cpu_and_shapes():
    task = RepeatTask(seq_len=4, n_symbols=10)
    m = SpikingHebbianLM(task.vocab)
    x, _, _ = task.batch(2)
    logits, sr = m(x, return_stats=True)
    assert logits.shape == (2, x.shape[1], task.vocab)
    assert 0.0 <= sr.item() <= 1.0


if __name__ == "__main__":   # runnable without pytest
    test_memory_is_necessary(); print("test_memory_is_necessary  PASS")
    test_runs_on_cpu_and_shapes(); print("test_runs_on_cpu_and_shapes  PASS")
