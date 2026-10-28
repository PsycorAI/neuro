# psymem/hebb.py -- 2026-10-27 (Neuro 85): encoder step 0 -- learn Psymem's semantic expansion layer with a local Hebbian
# rule instead of random FlyHash wiring. Rule = competing hidden units (Krotov & Hopfield 2019; BioHash, Ryali et al. 2020):
# for each input x (mean-centred, unit-norm bge vector) the most driven neuron moves toward x (Hebbian, Oja-normalized) and
# the rank-r neuron moves away (anti-Hebbian, -delta). Neurons start on random training turns ("exemplar cells").
# Usage: python psymem/hebb.py N_CORPUS M EPOCHS [init=exemplar|random]  -> models/psymem/hebb_{M}_e{EPOCHS}_{init}.npz (P, mu)
import os, sys, time
import numpy as np, torch

D = '/home/glenn/projects/neuro/'

if __name__ == '__main__':
    n, M, epochs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    init = sys.argv[4] if len(sys.argv) > 4 else 'exemplar'
    r, delta, B, lr0 = 7, 0.4, 2048, 0.02
    E = torch.tensor(np.load(D + f'data/psymem_enc/emb_{n}.npy'), dtype=torch.float32, device='cuda')
    mu = E.mean(0)
    X = torch.nn.functional.normalize(E - mu, dim=1)
    g = torch.Generator(device='cuda').manual_seed(0)
    W = X[torch.randperm(n, generator=g, device='cuda')[:M]].clone() if init == 'exemplar' else \
        torch.nn.functional.normalize(torch.randn(M, X.shape[1], generator=g, device='cuda'), dim=1)
    t0, steps = time.time(), epochs * (n // B)
    for step in range(steps):
        x = X[torch.randint(0, n, (B,), generator=g, device='cuda')]
        I = x @ W.T                                        # input currents (B x M)
        top = I.topk(r, dim=1).indices
        G = torch.zeros_like(I)
        G.scatter_(1, top[:, :1], 1.); G.scatter_(1, top[:, r - 1:], -delta)
        dW = G.T @ x - (G * I).sum(0)[:, None] * W        # Oja-style: g (x - (w.x) w)
        W += lr0 * (1 - step / steps) * dW / dW.abs().max().clamp_min(1e-12)
        if step % 500 == 0 or step == steps - 1:
            used = torch.unique(top[:, 0]).numel()
            print(f'step {step}/{steps} winners in batch {used} |W| {W.norm(dim=1).mean():.3f} {time.time() - t0:.0f}s', flush=True)
    os.makedirs(D + 'models/psymem', exist_ok=True)
    out = D + f'models/psymem/hebb_{M}_e{epochs}_{init}.npz'
    np.savez(out, P=W.cpu().numpy().astype(np.float32), mu=mu.cpu().numpy())
    print('saved', out)
