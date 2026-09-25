"""Train a scalar head on a FROZEN post-trained checkpoint.

  head_type: confidence   (idea #3) -- P(the model will get this prompt right),
      read at the last prompt position. Labels are generated ON POLICY inside the
      loop: greedy-decode the frozen model, compare to the gold tool call, and
      use its own observed correctness as the target. A head trained on someone
      else's notion of difficulty would not be calibrated to THIS model.

  head_type: energy       (idea #6) -- E(prompt, response), low = compatible.
      Trained with a margin ranking loss on the (chosen, rejected) pairs we
      already generate for ORPO, so it costs no new data. Use it for best-of-N
      reranking, to steer self_refine, and as the abstain threshold for #2.

The backbone is frozen throughout (eval mode, requires_grad False), so this can
never destabilise the model -- and a bad head is deleted rather than retrained.

  python src/train_heads.py --config configs/head_energy_v3_363m.yaml --device cuda
"""
import os, sys, glob, json, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import heads as H
import tool_data
from train_sft import build_model, make_encoder_pair
from train_dpo import load_preferences_jsonl, load_preferences

CKPT_ROOT = "/home/glenn/projects/neuro/models/checkpoints"
PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"


def resolve_ckpt(spec):
    if "/" in spec:
        return spec
    cks = sorted(glob.glob(f"{CKPT_ROOT}/{spec}/step_*.pt"))
    if not cks:
        raise FileNotFoundError(f"no ckpt for {spec}")
    return cks[-1]


# --------------------------------------------------------------- energy data --
def build_energy_batches(c, encode, max_len):
    """(prompt, chosen, rejected) -> encoded (ids, response_mask) for both sides."""
    raw = []
    if c.get("head_jsonl"):
        raw += load_preferences_jsonl(c["head_jsonl"], c.get("head_pair_types"))
    if c.get("head_dataset") or not raw:
        raw += load_preferences(c.get("head_dataset",
                                      "HuggingFaceH4/ultrafeedback_binarized"),
                                c.get("head_split", "train_prefs"))
    from train_dpo import encode_pair
    out, skipped = [], 0
    for ex in raw:
        w = encode_pair(ex["prompt"], ex["chosen"], encode, max_len)
        l = encode_pair(ex["prompt"], ex["rejected"], encode, max_len)
        if w is None or l is None:
            skipped += 1
            continue
        out.append((w, l))
    print(f"  energy pairs: {len(out)} encoded ({skipped} skipped)")
    return out


def energy_collate(batch, device):
    """Pad chosen+rejected into one (2B, T) block so a single forward covers
    both sides -- halves the frozen-backbone cost per pair."""
    seqs = [b[0] for b in batch] + [b[1] for b in batch]
    T = max(len(s[0]) for s in seqs)
    ids = torch.zeros(len(seqs), T, dtype=torch.long)
    msk = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (x, m) in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        msk[i, :len(m)] = torch.tensor(m, dtype=torch.bool)
    return ids.to(device), msk.to(device)


# ----------------------------------------------------------- confidence data --
def build_conf_examples(c, encode, tools):
    pairs, bad = tool_data.load_tool_jsonl(
        c["head_tools_jsonl"], tools, split="train")
    out = []
    for p in pairs:
        ids = encode(p["prompt"])
        if 8 < len(ids) < c.get("head_max_len", 512) - 80:
            out.append({"ids": ids, "rec": p["rec"], "response": p["response"]})
    print(f"  confidence examples: {len(out)} (bad lines {bad})")
    return out


@torch.no_grad()
def greedy_response(model, ids, block_size, device, max_new, decode):
    cur = list(ids)
    gen = []
    for _ in range(max_new):
        x = torch.tensor([cur[-block_size:]], dtype=torch.long, device=device)
        tok = int(model(x)[0, -1].argmax())
        gen.append(tok); cur.append(tok)
        if tool_data.EOS_MARK in decode(gen[-4:]):
            break
    return decode(gen)


def is_correct(text, rec):
    """Did the frozen model get this example right? Exact on tool calls (we hold
    the gold), mode-only on ANSWER examples (there is no single right wording)."""
    mode, call, _ = tool_data.parse_response(text)
    if rec["target_mode"] == tool_data.MODE_TOOL:
        g = rec["target_call"]
        return bool(call and call.get("name") == g["name"]
                    and call.get("arguments") == g["arguments"])
    return mode == tool_data.MODE_ANSWER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    device = args.device
    torch.manual_seed(c.get("seed", 0))
    kind = c["head_type"]
    assert kind in ("confidence", "energy"), kind

    # ---- frozen backbone --------------------------------------------------
    ck = resolve_ckpt(c["head_init"])
    blob = torch.load(ck, map_location=device, weights_only=False)
    model = build_model(c).to(device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_mem = c["d_mem"]
    print(f"frozen backbone: {ck} (step {blob.get('step')}) | d_mem={d_mem}")

    head = H.ScalarHead(d_mem).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=float(c.get("head_lr", 1e-3)),
                            weight_decay=float(c.get("head_wd", 0.01)))
    print(f"{kind} head: {n_head/1e3:.1f}k params, lr={c.get('head_lr', 1e-3)}")

    encode, decode = make_encoder_pair(c)
    max_len = c.get("head_max_len", c["block_size"])
    amp = c.get("amp", True) and device == "cuda"

    outdir = f"{CKPT_ROOT}/{c['run_name']}"
    os.makedirs(outdir, exist_ok=True)
    out_path = f"{outdir}/{kind}_head.pt"

    if kind == "energy":
        data = build_energy_batches(c, encode, max_len)
    else:
        data = build_conf_examples(c, encode, tool_data.load_tools())
    if not data:
        raise RuntimeError("no training data for the head")

    bs = c.get("head_batch", 8)
    steps = c.get("head_steps", 2000)
    margin = float(c.get("energy_margin", 1.0))
    rng = np.random.default_rng(c.get("seed", 0))
    t0 = time.time()
    run_loss, run_acc, seen = 0.0, 0.0, 0

    with H.HiddenCapture(model) as cap:
        for step in range(1, steps + 1):
            pick = rng.integers(0, len(data), size=bs)
            if kind == "energy":
                ids, msk = energy_collate([data[i] for i in pick], device)
                cap.clear()
                with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                    model(ids)
                h = cap.hidden.float()
                pooled = H.masked_mean_hidden(h, msk)
                e = head(pooled)
                B = len(pick)
                e_w, e_l = e[:B], e[B:]
                loss = H.energy_loss(e_w, e_l, margin)
                acc = (e_w < e_l).float().mean().item()
            else:
                rows = [data[i] for i in pick]
                labels, hs = [], []
                for r in rows:
                    text = greedy_response(model, r["ids"], c["block_size"],
                                           device, c.get("head_max_new", 64),
                                           decode)
                    labels.append(is_correct(text, r["rec"]))
                    x = torch.tensor([r["ids"]], dtype=torch.long, device=device)
                    cap.clear()
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                        model(x)
                    hs.append(cap.hidden.float()[0, -1])
                pooled = torch.stack(hs)
                logit = head(pooled)
                y = torch.tensor(labels, dtype=torch.float32, device=device)
                loss = H.confidence_loss(logit, y)
                acc = ((logit > 0).float() == y).float().mean().item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            run_loss += loss.item(); run_acc += acc; seen += 1

            if step % c.get("log_every", 50) == 0:
                print(f"step {step}/{steps} | loss {run_loss/seen:.4f} | "
                      f"{'pair_acc' if kind == 'energy' else 'label_acc'} "
                      f"{run_acc/seen:.3f} | {(time.time()-t0)/60:.1f} min",
                      flush=True)
                run_loss, run_acc, seen = 0.0, 0.0, 0
            if step % c.get("ckpt_every", 500) == 0 or step == steps:
                H.save_head(out_path, head, kind, c["head_init"], d_mem,
                            {"step": step, "margin": margin})
                print(f"  saved {out_path}", flush=True)

    print(f"done: {steps} steps in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
