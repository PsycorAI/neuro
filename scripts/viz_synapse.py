"""G4 - the money shot: show working memory forming IN CONTEXT.

We train the spiking Hebbian LM on the induction task, then feed one crafted
sequence whose bigrams repeat. We record, per timestep:
  (a) ||M|| - the synaptic memory growing as associations are written
  (b) P(correct successor) - prediction confidence, which JUMPS on the repeat
      because the synapse that binds that pair has been strengthened.
Saves assets/synapse_strengthening.png.
"""
import os, sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from data import RepeatTask

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "synapse_strengthening.png")


def quick_train(task, steps=800):
    torch.manual_seed(0)
    m = SpikingHebbianLM(task.vocab, d=128, n_neurons=256, d_mem=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(steps):
        x, y, _ = task.batch(128)
        loss = F.cross_entropy(m(x).reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def trace(model, seq):
    """Replay a single sequence, returning per-step ||M|| and P(true next token)."""
    model.eval()
    idx = torch.tensor(seq)[None]
    cur = model.to_current(model.embed(idx))
    B = 1
    mem = torch.zeros(B, model.n_neurons)
    M = torch.zeros(B, model.d_mem, model.d_mem)
    prev = torch.zeros(B, model.n_neurons)
    mnorm, pcorrect = [], []
    for t in range(len(seq) - 1):
        spk, mem = model.lif(cur[:, t, :], mem)
        k, v, q = model.W_k(prev), model.W_v(spk), model.W_q(spk)
        M = model.lam * M + model.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
        p = F.softmax(model.head(model.norm(r + model.W_ff(spk))), -1)[0]
        mnorm.append(M.norm().item())
        pcorrect.append(p[seq[t + 1]].item())
        prev = spk
    return mnorm, pcorrect


def main():
    task = RepeatTask(seq_len=8, n_symbols=20)
    model = quick_train(task)
    # one episode: a sequence, separator, then the same sequence again
    s = [3, 7, 1, 7, 12, 5, 7, 9]
    seq = s + [task.sep] + s
    mnorm, pcorrect = trace(model, seq)
    sep_pos = len(s)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(mnorm, "-o", color="#b5651d")
    ax[0].set_ylabel("||M||  (synaptic memory)")
    ax[0].set_title("Working memory forms in-context: synapses store, then recall")
    ax[1].plot(pcorrect, "-o", color="#2a6f97")
    ax[1].axhline(1 / task.S, ls="--", color="gray", label="chance")
    ax[1].set_ylabel("P(correct next token)")
    ax[1].set_xlabel("timestep")
    for a in ax:
        a.axvline(sep_pos, ls=":", color="red")
        a.text(sep_pos + 0.1, a.get_ylim()[1] * 0.9, "repeat begins", color="red", fontsize=9)
    ax[1].legend(loc="center left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {os.path.abspath(OUT)}")
    print(f"P(correct) first copy mean={sum(pcorrect[:sep_pos])/sep_pos:.3f}  "
          f"second copy mean={sum(pcorrect[sep_pos:])/(len(pcorrect)-sep_pos):.3f}")


if __name__ == "__main__":
    main()
