"""Subword-fallback encoder for the post-training stages.

PROBLEM. Our student vocab is the top-32767 Llama-3 tokens by frequency in the
pretraining corpus; the other 74.5% of Llama-3's vocab maps to id 0 (UNK).
train_sft.load_lut_encode() DROPS those tokens outright, so any word Llama-3
happens to tokenize as one OOV token vanishes from the text entirely:

    "TOOL {\"name\": \"calculator\", \"arguments\": {\"expression\": \"0.17 * 240\"}}"
 -> "TOOL {\"name\": \"\", \"\": {\"\": \"0.17 * 240"

That silently corrupts any dataset with vocabulary outside the pretraining head
-- fatal for tool calling, where names and argument values are arbitrary strings
("kilometers" came back as "ometers", "celsius" as "c").

FIX. Keep the pretraining tokenization for every token the student HAS, and
re-spell only the OOV ones out of in-vocab subwords (greedy longest match).
The student saw those subwords constantly in other words, so it can both read
and emit them; previously it only ever saw id 0 there.

This is opt-in (`lut_subword_fallback: true`) so existing runs and checkpoints
keep their exact old behaviour.
"""
import functools
import numpy as np

TOKENIZER_ID = "NousResearch/Meta-Llama-3-8B"
LUT_FMT = "/home/glenn/projects/neuro/data/vocab_map_{}.npy"
MAX_PIECE = 24          # longest in-vocab string we try to match greedily


@functools.lru_cache(maxsize=4)
def _tables(vocab):
    """(lut, inv, piece->student_id dict). Cached: building the string table
    decodes 32768 tokens one at a time."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    lut = np.load(LUT_FMT.format(vocab))
    inv = np.zeros(vocab, dtype=np.int64)
    kept = np.where(lut > 0)[0]
    inv[lut[kept]] = kept
    pieces = {}
    for sid in range(1, vocab):
        s = tk.decode([int(inv[sid])])
        # Skip tokens that are not a clean standalone string: partial UTF-8
        # fragments decode to U+FFFD and must never be emitted by the speller.
        if not s or "�" in s:
            continue
        # First writer wins, so shorter/more frequent ids are preferred for the
        # same surface form (student ids are frequency-ordered).
        pieces.setdefault(s, sid)
    return tk, lut, inv, pieces


def spell(text, vocab):
    """Encode arbitrary text using ONLY in-vocab pieces, greedy longest match.
    Returns (ids, n_unspellable_chars)."""
    _, _, _, pieces = _tables(vocab)
    out, i, dropped = [], 0, 0
    n = len(text)
    while i < n:
        hit = None
        for j in range(min(n, i + MAX_PIECE), i, -1):
            sid = pieces.get(text[i:j])
            if sid is not None:
                hit = (sid, j)
                break
        if hit is None:
            dropped += 1
            i += 1
            continue
        out.append(hit[0])
        i = hit[1]
    return out, dropped


def make_encoder(vocab, subword_fallback=True, report=False):
    """text -> list[int] student ids.

    subword_fallback=False reproduces train_sft.load_lut_encode exactly
    (OOV tokens dropped), so a config can opt out and stay bit-compatible.
    """
    tk, lut, _, _ = _tables(vocab)
    stats = {"tokens": 0, "oov": 0, "respelled": 0, "dropped_chars": 0}

    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        out = []
        for t in ids:
            stats["tokens"] += 1
            s = int(lut[t])
            if s != 0:
                out.append(s)
                continue
            stats["oov"] += 1
            if not subword_fallback:
                continue
            piece = tk.decode([t])
            sub, dropped = spell(piece, vocab)
            stats["dropped_chars"] += dropped
            if sub:
                stats["respelled"] += 1
                out.extend(sub)
        return out

    encode.stats = stats
    if report:
        encode.report = lambda: (
            f"[lut_spell] {stats['tokens']} tokens, {stats['oov']} OOV "
            f"({100*stats['oov']/max(1,stats['tokens']):.1f}%), "
            f"{stats['respelled']} re-spelled, "
            f"{stats['dropped_chars']} chars unspellable")
    return encode


def make_decoder(vocab):
    tk, _, inv, _ = _tables(vocab)

    def decode(ids):
        return tk.decode([int(inv[i]) for i in ids if 0 < i < vocab])
    return decode
