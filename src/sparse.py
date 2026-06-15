"""SET: Sparse Evolutionary Training (Mocanu et al., Nat. Commun. 2018).

Periodically prune the weakest-magnitude active synapses and regrow the same
number at random. Over training the connectivity self-organizes from a random
(Erdos-Renyi) graph toward a heavy-tailed / scale-free topology.
"""
import torch


class SET:
    def __init__(self, mask, zeta=0.3):
        self.mask = mask          # model's rec_mask buffer, edited in place
        self.zeta = zeta

    @torch.no_grad()
    def step(self, weight):
        active = self.mask.bool()
        n = int(active.sum())
        k = int(self.zeta * n)
        if k == 0:
            return
        thresh = torch.kthvalue(weight.abs()[active], k).values
        prune = active & (weight.abs() <= thresh)
        npr = int(prune.sum())
        self.mask[prune] = 0.0
        free = self.mask == 0
        free.fill_diagonal_(False)
        idx = free.nonzero(as_tuple=False)
        if idx.size(0) >= npr and npr > 0:
            sel = idx[torch.randperm(idx.size(0))[:npr]]
            self.mask[sel[:, 0], sel[:, 1]] = 1.0
