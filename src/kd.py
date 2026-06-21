"""Knowledge distillation from a Llama-3 teacher.

Procedure (per training step):
  1. Take a batch of input tokens. We have them in two forms:
       - student form (B, T) with our 16,384-vocab IDs
       - teacher form (B, T) with original Llama-3 IDs (128,256-vocab)
  2. Run teacher forward in no_grad with bf16 → teacher_logits (B, T, 128256).
  3. Project teacher distribution down to our 16K vocab by aggregating
     probabilities: teacher_probs_16k[i] = sum over Llama-3 IDs j where lut[j]=i.
  4. KD loss = soft cross-entropy between student logits and teacher_probs_16k.
     Standard hard-target CE is added with weight (1-alpha).

Vocab projection:
  lut: np.ndarray of shape (128256,), values in 0..16383.
  For each Llama-3 ID j, lut[j] tells us which student ID it maps to.
  IDs not in our top-16383 get mapped to 0 (UNK).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F


def load_lut(student_vocab=16384):
    """Load the Llama-3 → student vocab projection LUT as a torch LongTensor."""
    path = f"/home/glenn/projects/neuro/data/vocab_map_{student_vocab}.npy"
    return torch.from_numpy(np.load(path)).long()


def load_teacher(model_id="meta-llama/Llama-3.2-1B", device="cuda", dtype=torch.bfloat16):
    """Load a Llama teacher. Returns the model in eval mode, gradients disabled."""
    from transformers import AutoModelForCausalLM
    t = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    t = t.to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def teacher_distribution(teacher, raw_ids, lut, student_vocab=16384,
                        temperature=1.0, topk=128):
    """Run teacher forward on raw Llama-3 IDs, return projected probabilities.

    raw_ids: (B, T) original Llama-3 IDs
    lut:     (128256,) projection LongTensor on the same device
    topk:    Use top-K teacher tokens per position (saves ~99% memory vs full softmax).
             128 retains essentially all mass; larger K diminishing returns.
    Returns: (B, T, student_vocab) probability distribution (bf16 to save memory)
    """
    out = teacher(raw_ids).logits  # (B, T, 128256) in teacher dtype (bf16)
    if temperature != 1.0:
        out = out / temperature
    # Top-K to avoid materializing a full 128K-dim softmax tensor.
    # The truncated softmax is re-normalized; the tail mass dropped is typically <1%.
    topk_vals, topk_idx = out.topk(topk, dim=-1)        # (B, T, K)
    topk_probs = topk_vals.softmax(dim=-1)              # stays in teacher dtype
    # Project: scatter each top-K probability into its corresponding student slot
    B, T, _ = topk_probs.shape
    proj = torch.zeros(B, T, student_vocab,
                       device=topk_probs.device, dtype=topk_probs.dtype)
    student_idx = lut[topk_idx]                         # (B, T, K)
    proj.scatter_add_(2, student_idx, topk_probs)
    return proj


def kd_loss(student_logits, teacher_probs, alpha=0.5, target_ids=None,
            temperature=1.0):
    """Combined KD + hard-target loss.

    student_logits: (B, T, V)
    teacher_probs:  (B, T, V) — the projected soft targets (already softmaxed)
    target_ids:     (B, T) — hard next-token labels for CE term
    alpha:          weight on KD vs hard CE (alpha=1.0 = pure KD)
    Returns: scalar loss
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # Soft cross-entropy: -sum(teacher * log student) ; equivalent to KL up to const
    soft_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    # Temperature scaling correction (Hinton et al.)
    soft_loss = soft_loss * (temperature ** 2)
    if alpha >= 1.0 or target_ids is None:
        return soft_loss
    hard_loss = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)),
                                 target_ids.reshape(-1))
    return alpha * soft_loss + (1 - alpha) * hard_loss
