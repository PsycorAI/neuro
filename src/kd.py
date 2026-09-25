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


def load_teacher(model_id="meta-llama/Llama-3.2-1B", device="cuda", dtype=torch.bfloat16,
                  compile=False, compile_mode="reduce-overhead", int8=False):
    """Load a Llama teacher. Returns the model in eval mode, gradients disabled.

    compile=True wraps the model in torch.compile. The teacher is a plain HF
    transformer under fixed shapes (config batch_size × block_size) with
    no_grad -- the ideal torch.compile target. At 500M scale the teacher
    forward dominates step FLOPs (Phi-3.5-mini ~= 3.8B params, ~7-8x the
    student's fwd+bwd combined), so this is the largest end-to-end lever
    if it holds. reduce-overhead mode uses CUDA graphs (big wins for
    repeated fixed-shape calls); fall back to compile_mode="default" if
    the CUDA-graphs path complains about the KV-cache path or HF's
    output-dataclass boundary.
    """
    from transformers import AutoModelForCausalLM
    if int8 and device == "cuda":
        # bitsandbytes 8-bit quantization: halves teacher memory, speeds up
        # matmuls. Linear layers become Linear8bitLt (int8 storage, fp16
        # compute internally regardless of input dtype). Small distribution
        # drift in soft targets (typically ~1-2%) -- verify with one eval
        # after enabling. device_map handles placement internally, so no
        # subsequent .to(device) (would error on bnb-quantized weights).
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_8bit=True)
        t = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qc,
                                                  device_map={"": device})
        print(f"KD teacher loaded in int8 (bitsandbytes)")
    else:
        t = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        t = t.to(device)
    t = t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    if compile and device == "cuda":
        t = torch.compile(t, mode=compile_mode)
        print(f"KD teacher compiled: mode={compile_mode}")
    return t


@torch.no_grad()
def teacher_distribution(teacher, raw_ids, lut, student_vocab=16384,
                        temperature=1.0, topk=128):
    """Run teacher forward on raw Llama-3 IDs, return top-K teacher probs + their
    projected student-vocab indices (NOT a dense (B,T,student_vocab) distribution
    -- kd_loss gathers directly from these, see its docstring for why that's
    mathematically identical to the old scatter-into-dense-then-dot approach,
    without the alloc/scatter/full-vocab-multiply every microstep).

    raw_ids: (B, T) original Llama-3 IDs
    lut:     (128256,) projection LongTensor on the same device
    topk:    Use top-K teacher tokens per position (saves ~99% memory vs full softmax).
             128 retains essentially all mass; larger K diminishing returns.
    Returns: (topk_probs (B,T,K), student_idx (B,T,K))
    """
    out = teacher(raw_ids).logits  # (B, T, 128256) in teacher dtype (bf16)
    if temperature != 1.0:
        out = out / temperature
    # Top-K to avoid materializing a full 128K-dim softmax tensor.
    # The truncated softmax is re-normalized; the tail mass dropped is typically <1%.
    topk_vals, topk_idx = out.topk(topk, dim=-1)        # (B, T, K)
    topk_probs = topk_vals.softmax(dim=-1)              # stays in teacher dtype
    student_idx = lut[topk_idx]                         # (B, T, K)
    return topk_probs, student_idx


def kd_loss(student_logits, teacher_topk, alpha=0.5, target_ids=None,
            temperature=1.0):
    """Combined KD + hard-target loss.

    student_logits: (B, T, V)
    teacher_topk:    (topk_probs (B,T,K), student_idx (B,T,K)) from teacher_distribution
    target_ids:     (B, T) — hard next-token labels for CE term
    alpha:          weight on KD vs hard CE (alpha=1.0 = pure KD)
    Returns: scalar loss

    Soft-CE via gather, not scatter-then-dot: for the dense projection proj[v] =
    sum_{i: student_idx_i=v} topk_probs_i, sum_v proj[v]*log_probs[v] regroups to
    sum_i topk_probs_i * log_probs[student_idx_i] -- exactly what gather computes,
    since duplicate student_idx entries (multiple teacher tokens collapsing to the
    same UNK/merged slot) contribute the same log_probs[v] regardless of grouping.
    Bit-identical result (verified numerically), no (B,T,V) alloc/scatter/full-
    vocab multiply.
    """
    topk_probs, student_idx = teacher_topk
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    gathered = student_log_probs.gather(-1, student_idx)          # (B,T,K)
    soft_loss = -(topk_probs * gathered).sum(dim=-1).mean()
    # Temperature scaling correction (Hinton et al.)
    soft_loss = soft_loss * (temperature ** 2)
    if alpha >= 1.0 or target_ids is None:
        return soft_loss
    hard_loss = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)),
                                 target_ids.reshape(-1))
    return alpha * soft_loss + (1 - alpha) * hard_loss
