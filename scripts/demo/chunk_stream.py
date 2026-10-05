"""chunk_stream.py -- 2026-10-05 (r25, demo idea I). Streaming inference for chunked-attention models: attention sees
the current chunk only, the memory carries everything older -- the "remembers beyond its context window" demo.

ChunkStreamer.logits(ids) returns the next-token logits after `ids`, any length. Complete chunks older than the
previous one are folded into a carried memory state (cached, so a chat only pays for new chunks). The forward covers
the previous complete chunk + the current partial chunk with the state from before them: chunk boundaries stay aligned
with training, attention never crosses one, and the LIF membrane / Engram n-grams have a full chunk of warm-up before
the predicted position (they restart at every forward call). Cost: <= 2 chunks per token.
Matches a single chunked forward over the whole history up to the small fold-boundary transients (temp/test_chunk_stream.py)."""
import torch


class ChunkStreamer:
    def __init__(self, model, chunk):
        self.m, self.C = model, int(chunk)
        for b in model.blocks:
            if hasattr(b, "qkv"):
                b.attn_chunk = self.C
        self.cache = []                 # cache[i] = (ids of folded chunks 0..i, state after folding chunk i)

    def _state(self, ids, n_fold, ablate):
        """Memory state after the first n_fold complete chunks of ids (None if n_fold == 0)."""
        C, keep = self.C, 0
        while keep < min(len(self.cache), n_fold) and self.cache[keep][0] == ids[:(keep + 1) * C] \
                and self.cache[keep][2] == ablate:
            keep += 1
        del self.cache[keep:]
        state = self.cache[-1][1] if self.cache else None
        for i in range(keep, n_fold):
            x = torch.tensor([ids[i * C:(i + 1) * C]], device=self.m.head.weight.device)
            out = self.m(x, initial_state=state, return_final_state=True, ablate_memory=ablate)
            state = out[1] if isinstance(out, tuple) else out
            self.cache.append((ids[:(i + 1) * C], state, ablate))
        return state

    @torch.no_grad()
    def logits(self, ids, ablate=False):
        """Next-token logits (V,) after ids. Returns (logits, start) where start = first index of the forwarded window."""
        C = self.C
        n_complete = len(ids) // C
        cur = n_complete if len(ids) % C else n_complete - 1          # index of the chunk holding the last token
        start = max(0, cur - 1) * C                                    # warm-up = the previous complete chunk
        state = self._state(ids, start // C, ablate)
        x = torch.tensor([ids[start:]], device=self.m.head.weight.device)
        out = self.m(x, initial_state=state, ablate_memory=ablate) if state is not None else \
            self.m(x, ablate_memory=ablate)
        lg = out[0] if isinstance(out, tuple) else out
        return lg[0, -1], start
