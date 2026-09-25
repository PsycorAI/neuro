"""Two small heads bolted onto a FROZEN post-trained checkpoint.

  ConfidenceHead  (idea #3)  -- Cactus Needle ships "a calibrated confidence
      score from a learned head" so the harness can route: act, confirm, or
      refuse. Read at the LAST PROMPT POSITION, i.e. before the model commits to
      anything, so the routing decision can be made instead of the action.

  EnergyHead      (idea #6)  -- the cheap slice of Energy-Based Transformers:
      learn to VERIFY compatibility between a prompt and a candidate response
      (low energy = good match), then use it for best-of-N reranking, to steer
      scripts/self_refine.py, and as the abstain threshold for idea #2. No
      backbone retraining, so it cannot destabilise a model that took two weeks
      to pretrain.

Both are trained against the frozen backbone, so they are cheap, reversible, and
independent of the SFT/ORPO/OPD stages.

HIDDEN STATES WITHOUT TOUCHING model.py. `SpikingHebbianLM.head` is the LM head
and its input is exactly the final hidden state (model.py:1374). A forward
PRE-hook on that module captures it, so src/model.py -- the module the live
flagship_v3 pretrain imports -- stays untouched. The head is also called for the
MTP path a line later; we keep the FIRST capture per forward, which is the main
one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenCapture:
    """Context manager capturing the final hidden state fed to the LM head.

        with HiddenCapture(model) as cap:
            cap.clear(); model(x); h = cap.hidden     # (B, T, d_mem)
    """

    def __init__(self, model):
        self.model = model
        self.hidden = None
        self._handle = None

    def _hook(self, _module, args):
        if self.hidden is None and args:
            self.hidden = args[0]
        return None

    def clear(self):
        self.hidden = None

    def __enter__(self):
        self._handle = self.model.head.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        return False


class ScalarHead(nn.Module):
    """Bottlenecked scalar head. Shared by both uses -- the difference is what
    hidden state you feed it and what loss you train it with, not the shape."""

    def __init__(self, d, bottleneck=None):
        super().__init__()
        h = bottleneck or max(64, d // 4)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, 1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x)))).squeeze(-1)


def last_prompt_hidden(hidden, prompt_lens):
    """(B, T, d) + (B,) -> (B, d) at each sequence's own last prompt position.
    That position's representation is causally determined by the prompt alone,
    which is what makes a PRE-action routing signal possible."""
    idx = (prompt_lens - 1).clamp(min=0)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def masked_mean_hidden(hidden, mask):
    """(B, T, d) + (B, T) bool -> (B, d) mean over the masked positions."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


def confidence_loss(logit, correct):
    """BCE against an on-policy correctness label. Calibration comes from the
    label being the model's OWN observed correctness, not a human guess."""
    return F.binary_cross_entropy_with_logits(logit.float(), correct.float())


def energy_loss(e_chosen, e_rejected, margin=1.0):
    """Margin ranking: chosen must sit at least `margin` BELOW rejected.
    Hinge rather than logistic so pairs already ordered by more than the margin
    stop contributing gradient -- the head should spend its capacity on the
    pairs it still gets wrong."""
    return F.relu(margin + e_chosen - e_rejected).mean()


def save_head(path, head, kind, base, d, extra=None):
    torch.save({"head": head.state_dict(), "kind": kind, "base": base,
                "d": d, **(extra or {})}, path)


def load_head(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    head = ScalarHead(blob["d"]).to(device)
    head.load_state_dict(blob["head"])
    head.eval()
    return head, blob
