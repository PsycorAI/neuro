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


# ---- 2026-10-22 (Neuro 73, Glenn OK: v4f throughput lever #1): the teacher in ANOTHER process / GPU -----------------
# The KD phase runs with only the 5090 visible; the teacher moves to the 5080 as a server (temp/kd_teacher_server.py)
# that computes teacher_distribution() exactly as the in-process path does (same compile, under bf16 autocast) and
# returns (top-K probs fp32, student idx int16) over a Unix socket. The trainer asks one micro-batch ahead, so the
# 5080 works on micro i+1 while the 5090 trains on micro i. Probs are sent as fp32: exact for bf16 or fp32 originals,
# and kd_loss promotes to fp32 anyway. Wire: request = int32 [magic, B, T] + B*T int32 raw ids;
# reply = B*T*K fp32 probs + B*T*K int16 student idx.
KD_MAGIC = 0x4B44_5431


def recv_exact(sock, view):
    n = 0
    while n < len(view):
        k = sock.recv_into(view[n:])
        if k == 0:
            raise ConnectionError("KD teacher server closed the connection")
        n += k


class RemoteTeacher:
    def __init__(self, path, B, T, topk=128, timeout=1800, connect_wait=900):
        import socket, time
        self.B, self.T, self.K = int(B), int(T), int(topk)
        t0 = time.time()
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                break
            except OSError:
                self.sock.close()
                if time.time() - t0 > connect_wait:
                    raise RuntimeError(f"no KD teacher server at {path} after {connect_wait} s")
                time.sleep(5)
        self.sock.settimeout(timeout)
        n = self.B * self.T * self.K
        # two pinned reply buffers, alternated; an event guards a buffer until its GPU copy has finished
        self.bufs = [torch.empty(n * 6, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.events, self.slot = [None, None], 0
        print(f"KD teacher: remote server {path} (B{self.B} T{self.T} top-{self.K})", flush=True)

    def request(self, raw_ids_cpu):
        assert tuple(raw_ids_cpu.shape) == (self.B, self.T), raw_ids_cpu.shape
        hdr = np.array([KD_MAGIC, self.B, self.T], dtype=np.int32).tobytes()
        self.sock.sendall(hdr + raw_ids_cpu.numpy().astype(np.int32).tobytes())

    def result(self, device):
        """Reply to the OLDEST outstanding request (FIFO) -> (topk_probs fp32, student_idx long) on `device`."""
        s = self.slot; self.slot ^= 1
        if self.events[s] is not None:
            self.events[s].synchronize()
        buf = self.bufs[s]
        recv_exact(self.sock, memoryview(buf.numpy()))
        n = self.B * self.T * self.K
        probs = buf[: 4 * n].view(torch.float32).view(self.B, self.T, self.K).to(device, non_blocking=True)
        idx = buf[4 * n:].view(torch.int16).view(self.B, self.T, self.K).to(device, non_blocking=True).long()
        ev = torch.cuda.Event(); ev.record(); self.events[s] = ev
        return probs, idx
