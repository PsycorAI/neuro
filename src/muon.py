"""Muon optimizer (Momentum Orthogonalized by Newton-Schulz).

Drop-in optimizer for 2D weight matrices (hidden-layer linear projections).
Empirically ~25-35% faster training to equal loss than AdamW at small scale,
state of the art on the NanoGPT speedrun benchmark in 2025-2026.

Use AdamW for embeddings, biases, LayerNorm scales -- anything that is not a
2D weight matrix between hidden layers.

Reference:
  Keller Jordan et al., "Muon: An optimizer for hidden layers in neural
  networks." https://kellerjordan.github.io/posts/muon/

This is a single-GPU implementation. The distributed variant is omitted.
"""
import torch


@torch.no_grad()
def _newton_schulz(G, steps=5, eps=1e-7):
    """Apply ~5 Newton-Schulz iterations to orthogonalize G in place.
    Coefficients tuned to converge fast in a few iterations (Keller Jordan).
    Operates in bfloat16 internally for speed; result cast back to G's dtype.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        # validate: every param must be a 2D matrix
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon is for 2D weight matrices only; got shape {tuple(p.shape)}. "
                        f"Pass 1D params (embeddings, biases, LayerNorm) to AdamW instead.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                # Nesterov lookahead
                update = g.add(buf, alpha=mom) if nesterov else buf
                # Orthogonalize via Newton-Schulz
                update = _newton_schulz(update, steps=ns)
                # LR scaled by sqrt(max(d_out, d_in) / min(d_out, d_in)) -- "muP-style"
                scale = max(1.0, (max(p.shape) / min(p.shape)) ** 0.5)
                p.add_(update, alpha=-lr * scale)
        return loss


def build_muon_adamw(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.0,
                    adamw_fused=True):
    """Split model parameters and return (muon, adamw) optimizers.

    2D matrices in hidden layers -> Muon.
    Everything else (embeddings, biases, LayerNorm, output head if you want it
    in AdamW) -> AdamW.

    Convention: the output head (lm_head / classifier) and the input embedding
    are kept in AdamW because they interact with vocabulary size.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() == 2
        is_embedding = "embed" in name.lower() or "pos" in name.lower()
        is_head = "head" in name.lower() or "lm_head" in name.lower()
        if is_matrix and not is_embedding and not is_head:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    if not muon_params:
        muon = None
    else:
        muon = Muon(muon_params, lr=muon_lr)
    adamw_kwargs = dict(lr=adamw_lr, weight_decay=weight_decay)
    if adamw_fused and torch.cuda.is_available():
        adamw_kwargs["fused"] = True
    adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs)
    return muon, adamw, len(muon_params), len(adamw_params)
