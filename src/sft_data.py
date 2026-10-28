"""Supervised fine-tuning data: Alpaca-format dataset → padded batches with
loss masking (CE on response tokens only; instruction tokens masked with -100).

Chat template (works with our 16k LUT vocab — no special tokens needed):
    ### Instruction:
    {instruction}
    [optional ### Input: {input}]
    ### Response:
    {response}<eos>

The dataset comes from HuggingFace (default: yahma/alpaca-cleaned, ~52k samples).
Output: list of (input_ids, labels) where labels are -100 on prompt tokens.
"""
import numpy as np
import torch

PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
EOS_MARK = "\n###"   # cheap end marker our LUT vocab can encode


def render_conversation(history, words=50):
    """2026-10-21 (Neuro 72, chat brain): earlier turns [[user, assistant], ...] as a block placed before the current
    turn's prompt, one line per message, assistant replies cut to `words` words. No ANSWER / TOOL words (the model copied
    them from history, n66). Shared by training (format_example, tool_data.format_prompt) and the demo engine."""
    if not history:
        return ""
    one = lambda s, n=10 ** 6: " ".join(str(s).split()[:n]) + (" ..." if len(str(s).split()) > n else "")
    return "### Conversation:\n" + "".join(f"User: {one(q)}\nAssistant: {one(a, words)}\n" for q, a in history) + "\n"


def format_example(ex):
    inp = ex.get("input", "").strip()
    if inp:
        prompt = PROMPT_WITH_INPUT.format(instruction=ex["instruction"].strip(),
                                           input=inp)
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=ex["instruction"].strip())
    response = ex["output"].strip() + EOS_MARK
    return render_conversation(ex.get("history")) + prompt, response


def encode_with_mask(prompt, response, encode_fn, max_len):
    """Returns (input_ids, labels). labels[t] = -100 for prompt tokens, else target id.
    Truncates at max_len. Drops samples where prompt alone exceeds max_len (no room for response)."""
    p_ids = encode_fn(prompt)
    r_ids = encode_fn(response)
    if not r_ids or len(p_ids) >= max_len - 1:
        return None
    full = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return full, labels


def load_alpaca(dataset="yahma/alpaca-cleaned", split="train"):
    """Load an SFT dataset. Auto-converts HuggingFace 'messages'-format
    datasets (Tulu-3, OpenAssistant, WildChat, ...) to Alpaca-style
    {instruction, input, output} at load time; native Alpaca datasets
    pass through unchanged. For multi-turn conversations, takes the FIRST
    user->assistant pair only (v1 simplification).
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    raw = list(ds)
    if raw and "messages" in raw[0]:
        converted = []
        for ex in raw:
            msgs = ex.get("messages", []) or []
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
            if user_msg and asst_msg and user_msg.get("content") and asst_msg.get("content"):
                converted.append({
                    "instruction": user_msg["content"].strip(),
                    "input": "",
                    "output": asst_msg["content"].strip(),
                })
        print(f"[sft_data] converted {len(converted)}/{len(raw)} messages-format examples to Alpaca fields")
        return converted
    return raw


def build_sft_examples(raw, encode_fn, max_len):
    """Encode + mask all examples. Returns list of (input_ids, labels)."""
    out = []; skipped = 0
    n = len(raw)
    reported = 0
    for i, ex in enumerate(raw):
        if n >= 1000:
            pct = 100 * (i + 1) // n
            if pct >= reported + 10:
                print(f"  encoding: {pct}% ({i+1}/{n})", flush=True)
                reported = pct
        try:
            prompt, response = format_example(ex)
            r = encode_with_mask(prompt, response, encode_fn, max_len)
            if r is None:
                skipped += 1; continue
            out.append(r)
        except Exception:
            skipped += 1
    return out, skipped


def collate(batch, pad_id=0):
    """Pad to longest in batch. Returns (input_ids, labels, attention_mask).
    Labels padded with -100 (ignore_index for cross_entropy)."""
    max_len = max(len(b[0]) for b in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    lbl = torch.full((len(batch), max_len), -100, dtype=torch.long)
    msk = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        n = len(x)
        ids[i, :n] = torch.tensor(x, dtype=torch.long)
        lbl[i, :n] = torch.tensor(y, dtype=torch.long)
        msk[i, :n] = True
    return ids, lbl, msk


class SFTLoader:
    """Random-shuffle iterator over the encoded dataset."""
    def __init__(self, examples, batch_size, seed=0):
        self.examples = examples
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(examples))
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor + self.batch_size > len(self.examples):
            self.order = self.rng.permutation(len(self.examples))
            self.cursor = 0
        batch_idx = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return collate([self.examples[i] for i in batch_idx])
