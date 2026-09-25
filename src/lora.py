"""LoRA (Low-Rank Adaptation) for parameter-efficient finetuning.

Wraps selected `nn.Linear` layers of an already-built model with a low-rank
update  W' x = W x + (alpha/r) * B (A x),  training only the tiny A/B factors
while the base weight W stays frozen. See Hu et al., arXiv:2106.09685.

Why this exists in Neuro
------------------------
The SFT/DPO/persona stages (train_sft.py, train_dpo.py) currently full-finetune
every parameter. LoRA makes them:
  - faster (a fraction of the params get gradients/optimizer state),
  - lower-VRAM (no full AdamW/Muon moment tensors for the frozen base),
  - and — the demo-relevant win — it turns a finetune into a tiny, swappable
    "skill adapter": a few MB of A/B tensors you can load onto the frozen base,
    stack, or merge back in for inference.

This is a DIFFERENT kind of "brain mod" than the fast-weight memory M
(scripts/brain_mod_composition.py): M is runtime EPISODIC memory (state that
lives in the delta-rule associative store); a LoRA adapter is a DURABLE LEARNED
skill baked into the weights. They compose — LoRA never touches the M state.

Design guarantees
-----------------
  * B is ZERO-initialized, so a freshly injected adapter is an EXACT no-op:
    output is bit-identical to the base model until the adapter is trained.
    Safe to drop onto any pretrained checkpoint.
  * The base Linear's weight/bias are frozen in place (requires_grad_(False));
    only lora_A / lora_B carry gradients.
  * Degenerate (out==1) matrices — the delta-rule gates W_beta/W_alpha/W_gate,
    Titans' W_theta/W_eta, MoR routers — are NEVER wrapped (a rank-r factor of a
    1-row matrix is pointless and would perturb the gates we deliberately keep
    in AdamW). Guarded by min(shape) > 1.
  * Adapter save/load is adapter-ONLY (lora_state_dict / load_lora_state_dict):
    the swappable-mod file, independent of the base checkpoint.
  * merge_lora() folds (alpha/r) * B@A into the base weight for zero-overhead
    inference / brain-mod bake, and unwraps back to plain nn.Linear.

Standalone module (no import side effects on the live trainer). Wiring into the
SFT/DPO trainers is additive and behind a default-off `lora_enabled` flag.
"""
import math
import torch
import torch.nn as nn


# Default targets: the delta-rule K/V/Q projections — the reasoning matmuls,
# the direct analog of attention's Q/K/V that standard LoRA adapts. Always plain
# nn.Linear (W_ff becomes a SwiGLU sub-module under use_swiglu, so it is opt-in
# by listing "W_ff"/"gate"/"up"/"down" explicitly rather than defaulted).
DEFAULT_TARGETS = ("W_q", "W_k", "W_v")


class LoRALinear(nn.Module):
    """Frozen base `nn.Linear` plus a trainable low-rank update.

    forward(x) = base(x) + scaling * (x @ A^T) @ B^T
      A: (r, in_features)      init: Kaiming-uniform (nonzero, gets gradient)
      B: (out_features, r)     init: zeros            -> no-op at construction
      scaling = alpha / r
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        assert r > 0
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> exact no-op until trained.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * delta

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def merged_linear(self) -> nn.Linear:
        """Return a plain nn.Linear with the adapter folded into the weight."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B @ self.lora_A)   # (out, in)
            merged.weight.copy_(self.base.weight + delta_w.to(self.base.weight.dtype))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device)


def _iter_target_linears(model, targets):
    """Yield (parent_module, attr_name, full_name, linear) for each wrappable
    nn.Linear whose dotted name contains any target substring."""
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.out_features == 1 or module.in_features == 1:
            continue   # degenerate gate/router matrix — never adapt
        if not any(t in full_name for t in targets):
            continue
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        yield parent, attr, full_name, module


def inject_lora(model, r=8, alpha=16, dropout=0.0, targets=DEFAULT_TARGETS):
    """Replace matching nn.Linear layers with LoRALinear, in place.

    Returns (n_wrapped, adapter_param_count, wrapped_names)."""
    wrapped_names = []
    n_params = 0
    for parent, attr, full_name, linear in _iter_target_linears(model, targets):
        lora = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora)
        wrapped_names.append(full_name)
        n_params += lora.lora_A.numel() + lora.lora_B.numel()
    if not wrapped_names:
        raise ValueError(
            f"inject_lora wrapped 0 layers for targets={targets}. "
            f"Check target substrings against model.named_modules().")
    return len(wrapped_names), n_params, wrapped_names


def mark_only_lora_trainable(model):
    """Freeze every parameter except LoRA A/B. Returns the trainable-param list."""
    trainable = []
    for name, p in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)
    return trainable


def lora_state_dict(model):
    """Adapter-only state dict (the swappable-mod file): just lora_A/lora_B."""
    return {k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if k.endswith("lora_A") or k.endswith("lora_B")}


def load_lora_state_dict(model, sd, strict=True):
    """Load an adapter-only state dict into an already-LoRA-injected model."""
    own = model.state_dict()
    missing = [k for k in own if (k.endswith("lora_A") or k.endswith("lora_B"))
               and k not in sd]
    unexpected = [k for k in sd if k not in own]
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA load mismatch: missing={missing[:3]} "
                           f"unexpected={unexpected[:3]}")
    with torch.no_grad():
        for k, v in sd.items():
            if k in own:
                own[k].copy_(v.to(own[k].device))
    return missing, unexpected


def merge_lora(model):
    """Fold every adapter into its base weight and unwrap back to nn.Linear,
    in place. For zero-overhead inference and brain-mod bake. Returns n_merged."""
    n = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, module.merged_linear())
        n += 1
    return n


# --------------------------------------------------------------------------- #
# CPU smoke test — no GPU or dataset needed:  python src/lora.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    class Toy(nn.Module):
        """Mimics the block's projections: adaptable W_q/W_k/W_v + a (N,1) gate
        that must be skipped + an unrelated Linear that must be skipped."""
        def __init__(self, d=64, dm=48):
            super().__init__()
            self.W_q = nn.Linear(d, dm, bias=False)
            self.W_k = nn.Linear(d, dm, bias=False)
            self.W_v = nn.Linear(d, dm, bias=False)
            self.W_gate = nn.Linear(dm, 1)      # degenerate -> must be skipped
            self.other = nn.Linear(d, dm)        # not a target -> must be skipped
            self.head = nn.Linear(dm, 100)

        def forward(self, x):
            h = self.W_q(x) + self.W_k(x) + self.W_v(x)
            return self.head(h)

    x = torch.randn(4, 10, 64)

    base = Toy()
    base.eval()
    ref = base(x).clone()

    # 1) injection wraps exactly the 3 targets, not gate/other/head
    n_wrapped, n_params, names = inject_lora(base, r=8, alpha=16)
    assert n_wrapped == 3, (n_wrapped, names)
    assert all(any(t in nm for t in ("W_q", "W_k", "W_v")) for nm in names), names
    assert not any("gate" in nm or "other" in nm or "head" in nm for nm in names), names

    # 2) exact no-op at init (B is zero) -> bit-identical output
    base.eval()
    assert torch.equal(base(x), ref), "LoRA must be a no-op at injection"

    # 3) only lora_A/lora_B are trainable after freezing
    trainable = mark_only_lora_trainable(base)
    tset = {id(p) for p in trainable}
    for nm, p in base.named_parameters():
        want = nm.endswith("lora_A") or nm.endswith("lora_B")
        assert p.requires_grad == want, nm
        assert (id(p) in tset) == want, nm
    n_all = sum(p.numel() for p in base.parameters())
    n_tr = sum(p.numel() for p in trainable)
    assert n_tr < 0.2 * n_all, (n_tr, n_all)

    # 4) a gradient step makes the adapter active (output moves)
    opt = torch.optim.SGD(trainable, lr=0.1)
    tgt = torch.randn(4, 10, 100)
    for _ in range(5):
        opt.zero_grad()
        loss = ((base(x) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    base.eval()
    trained_out = base(x).clone()
    assert not torch.allclose(trained_out, ref), "adapter should change output after training"

    # 5) adapter-only save is tiny and has only lora keys
    sd = lora_state_dict(base)
    assert sd and all(k.endswith("lora_A") or k.endswith("lora_B") for k in sd), sd.keys()
    assert len(sd) == 2 * n_wrapped, sd.keys()

    # 6) load adapter into a freshly injected model -> reproduces trained output
    fresh = Toy()
    fresh.load_state_dict({k: v for k, v in base.state_dict().items()
                           if not (k.endswith("lora_A") or k.endswith("lora_B"))},
                          strict=False)
    # ^ fresh still has the ORIGINAL base weights (Toy() re-init) — so copy base
    #   weights over to isolate the adapter effect:
    fresh = Toy()
    inject_lora(fresh, r=8, alpha=16)
    # copy frozen base weights from `base` so only the adapter differs
    with torch.no_grad():
        for (kn, kp), (fn, fp) in zip(base.named_parameters(), fresh.named_parameters()):
            if not (kn.endswith("lora_A") or kn.endswith("lora_B")):
                fp.copy_(kp)
    load_lora_state_dict(fresh, sd)
    fresh.eval()
    assert torch.allclose(fresh(x), trained_out, atol=1e-6), "adapter load must reproduce output"

    # 7) merge folds the adapter in and matches the wrapped output
    merged_out_before = base(x).clone()
    n_merged = merge_lora(base)
    assert n_merged == n_wrapped, n_merged
    assert not any(isinstance(m, LoRALinear) for m in base.modules()), "merge must unwrap"
    base.eval()
    assert torch.allclose(base(x), merged_out_before, atol=1e-5), "merge must preserve output"
    assert not lora_state_dict(base), "merged model has no adapter keys"

    print(f"[OK] lora.py: wrapped {n_wrapped} layers, {n_params} adapter params "
          f"({100*n_tr/n_all:.1f}% trainable), no-op@init, "
          f"train/save/load/merge all verified")
