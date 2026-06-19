"""Phase 2: SET on the neuron->neuron synapses.

Trains the recurrent spiking model with SET, then shows the synaptic graph
evolved toward a heavy-tailed degree distribution and is modular, while the
induction recall task stays solved. Saves assets/degree_distribution.png.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikingHebbianLM
from data import RepeatTask
from sparse import SET

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "degree_distribution.png")


def degree(mask):
    A = mask.cpu().numpy()
    return A.sum(0) + A.sum(1)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum() + 1e-9))


def main():
    torch.manual_seed(0)
    task = RepeatTask(8, 20)
    model = SpikingHebbianLM(task.vocab, recurrent=True, rec_density=0.02)
    blk = model.blocks[0]
    setm = SET(blk.rec_mask, zeta=0.3)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    deg0 = degree(blk.rec_mask).copy()

    t0 = time.time()
    for step in range(1, 2001):
        x, y, _ = task.batch(128)
        loss = F.cross_entropy(model(x).reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            setm.step(blk.W_rec.weight)

    x, y, second = task.batch(512)
    with torch.no_grad():
        acc = (model(x).argmax(-1)[:, second] == y[:, second]).float().mean().item()

    deg1 = degree(blk.rec_mask)
    A = ((blk.rec_mask.cpu().numpy() + blk.rec_mask.cpu().numpy().T) > 0).astype(float)
    G = nx.from_numpy_array(A)
    Q = nx.community.modularity(G, nx.community.greedy_modularity_communities(G))
    E = G.number_of_edges()
    Gr = nx.gnm_random_graph(blk.n_neurons, E, seed=0)
    Qr = nx.community.modularity(Gr, nx.community.greedy_modularity_communities(Gr))

    print(f"trained SET-recurrent in {time.time()-t0:.0f}s | recall_acc={acc:.3f}")
    print(f"degree Gini   init={gini(deg0):.3f} -> final={gini(deg1):.3f}")
    print(f"max/mean deg  init={deg0.max():.0f}/{deg0.mean():.1f} -> final={deg1.max():.0f}/{deg1.mean():.1f}")
    print(f"modularity Q  SET={Q:.3f}  vs random ER={Qr:.3f}")
    g_heavy = gini(deg1) > gini(deg0) + 0.05
    g_mod = Q > Qr
    g_task = acc > 0.8
    print(f"RESULT: heavy-tailed={'PASS' if g_heavy else 'CHECK'}  "
          f"modular={'PASS' if g_mod else 'CHECK'}  task-intact={'PASS' if g_task else 'FAIL'}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for d, lab, c in [(deg0, "initial (random)", "#999999"), (deg1, "after SET", "#b5651d")]:
        v, cnt = np.unique(d[d > 0], return_counts=True)
        ax[0].loglog(v, cnt, "o-", label=lab, color=c)
    ax[0].set_xlabel("degree"); ax[0].set_ylabel("count")
    ax[0].set_title("Degree distribution (log-log)"); ax[0].legend()
    ax[1].hist(deg1, bins=30, color="#2a6f97")
    ax[1].set_title(f"After SET: Gini={gini(deg1):.2f}, modularity Q={Q:.2f}")
    ax[1].set_xlabel("degree")
    fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True); fig.savefig(OUT, dpi=120)
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
