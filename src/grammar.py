"""Grammar-constrained decoding for the tool-call protocol (idea #4).

WHY. A constrained 3B beat an unconstrained 70B on structured output in the
2026 XGrammar-2 results, and OpenAI's own numbers go from <40% to 100% schema
compliance. For a 426M model this is not a nicety: unconstrained, most of its
probability mass at a structural position is spent on tokens that cannot
possibly continue a valid call. Masking them costs nothing and removes an entire
failure class (malformed JSON, invented tool names, missing/extra arguments).

HOW. The protocol is small enough to be an explicit character DFA rather than a
general JSON parser (see src/tool_data for the locked format):

    TOOL {"name": "<one of the offered tools>", "arguments": {"<p1>": <v1>, ...}}
    ANSWER <free text>

Node kinds:
  LIT   literal character branching (also the trie over tool names / enum values)
  STR   inside a quoted string value: any printable but '"' and '\\'; '"' exits
  NUM   a JSON number: [-0-9.]; the following ',' or '}' exits
  FREE  anything (ANSWER mode)
  END   the call is complete; only the EOS mark may follow
  ACC   accepted, generation stops

Token-level masks are derived by walking each vocab token's characters through
the DFA, and cached per node. A tool call visits a few dozen nodes, so only
those get built.

Argument ORDER is fixed to the schema order, which is what lets the DFA stay
this small -- and it is the same order src/tool_data.dumps_call emits, so the
training targets are inside the constrained language by construction.
"""
import numpy as np
import torch

from tool_data import (EOS_MARK, MODE_TOOL, MODE_ANSWER, load_tools,
                       tools_by_name)

LIT, STR, NUM, FREE, END, ACC = "LIT", "STR", "NUM", "FREE", "END", "ACC"
# Printable string content. Deliberately excludes '"' (it terminates the value)
# and '\\' (escape handling would double the DFA for no benefit) -- tool_data
# .sanitize_arg_values strips both from the training targets to match.
STR_CHARS = set(
    " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~")
NUM_CHARS = set("-0123456789.eE+")


class ToolGrammar:
    """DFA for one *offered tool set*. Build one per distinct offered set (they
    are cheap; the expensive part is the per-node token masks, which are cached
    on the shared TokenTable)."""

    def __init__(self, offered, tools=None):
        self.tools = tools or load_tools()
        self.by = tools_by_name(self.tools)
        self.offered = [n for n in offered if n in self.by]
        self.nodes = []
        self._build()

    # ---------------------------------------------------------- construction
    def _new(self, kind, **kw):
        self.nodes.append(dict(kind=kind, edges={}, **kw))
        return len(self.nodes) - 1

    def _lit(self, frm, text, to=None):
        """Chain `text` from node frm, returning the final node (created if
        needed). Shares prefixes when they already exist, which is what makes
        the tool-name alternation a trie."""
        cur = frm
        for i, ch in enumerate(text):
            last = (i == len(text) - 1)
            nxt = self.nodes[cur]["edges"].get(ch)
            if nxt is None:
                nxt = to if (last and to is not None) else self._new(LIT)
                self.nodes[cur]["edges"][ch] = nxt
            cur = nxt
        return cur

    def _build(self):
        self.root = self._new(LIT)
        self.acc = self._new(ACC)
        self.end = self._new(END)
        self._lit(self.end, EOS_MARK, self.acc)

        # ANSWER <free text>
        ans = self._new(FREE)
        self._lit(self.root, MODE_ANSWER + " ", ans)
        self.nodes[ans]["eos_from"] = self.end     # free text may stop any time

        # TOOL {"name": "<name>", "arguments": {...}}
        head = self._lit(self.root, MODE_TOOL + ' {"name": "')
        for name in self.offered:
            after = self._lit(head, name + '", "arguments": {')
            params = self.by[name]["params"]
            if not params:
                self._lit(after, "}}", self.end)
                continue
            cur = after
            for i, p in enumerate(params):
                tail = "}}" if i == len(params) - 1 else ", "
                cur = self._value_node(cur, p, tail, i == len(params) - 1)
            # cur is now the node reached after the final tail
        # (nothing else is reachable: any other continuation has no edge)

    def _value_node(self, cur, p, tail, is_last):
        """Emit '"<pname>": <value><tail>' and return the node after tail."""
        if p["type"] == "enum":
            cur = self._lit(cur, f'"{p["name"]}": ')
            outs = []
            for v in p["values"]:
                outs.append(self._lit(cur, f'"{v}"'))
            # all enum branches continue through the same tail
            shared = self.end if is_last else self._new(LIT)
            for o in outs:
                self._lit(o, tail, shared)
            return shared
        if p["type"] == "number":
            cur = self._lit(cur, f'"{p["name"]}": ')
            nxt = self.end if is_last else self._new(LIT)
            num = self._new(NUM)
            # entering the number: any NUM char moves into the NUM node. The
            # tail chars (',' / '}') are not in NUM_CHARS, so there is no
            # ambiguity about when the number ends.
            for ch in NUM_CHARS:
                self.nodes[cur]["edges"][ch] = num
            for ch in NUM_CHARS:
                self.nodes[num]["edges"][ch] = num
            self._lit(num, tail, nxt)
            return nxt
        # string
        cur = self._lit(cur, f'"{p["name"]}": "')
        s = self._new(STR)
        for ch in STR_CHARS:
            self.nodes[cur]["edges"][ch] = s
            self.nodes[s]["edges"][ch] = s
        closed = self._new(LIT)
        self.nodes[cur]["edges"]['"'] = closed     # empty string value
        self.nodes[s]["edges"]['"'] = closed
        nxt = self.end if is_last else self._new(LIT)
        self._lit(closed, tail, nxt)
        return nxt

    # ------------------------------------------------------------- stepping
    def step(self, node, ch):
        """One character. Returns the next node id, or None if `ch` is not
        allowed here."""
        n = self.nodes[node]
        if n["kind"] == FREE:
            return node
        if n["kind"] == ACC:
            return None
        return n["edges"].get(ch)

    def walk(self, node, text):
        for ch in text:
            node = self.step(node, ch)
            if node is None:
                return None
        return node

    def accepting(self, node):
        return self.nodes[node]["kind"] == ACC


class TokenTable:
    """student-id -> surface string, plus a cache of per-node token masks.
    Built once and shared across grammars (the string table costs ~12s)."""

    def __init__(self, vocab):
        import lut_spell
        tk, lut, inv, _ = lut_spell._tables(vocab)
        self.vocab = vocab
        self.strs = [""] * vocab
        for sid in range(1, vocab):
            s = tk.decode([int(inv[sid])])
            self.strs[sid] = "" if "�" in s else s
        self._cache = {}

    def mask(self, grammar, node, device="cpu"):
        """(bool mask over the vocab, int64 next-node per token)."""
        key = (id(grammar), node)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0].to(device), hit[1].to(device)
        if grammar.nodes[node]["kind"] == FREE:
            m = torch.ones(self.vocab, dtype=torch.bool)
            m[0] = False                      # id 0 is UNK, never emit it
            nx = torch.full((self.vocab,), node, dtype=torch.long)
        else:
            m = torch.zeros(self.vocab, dtype=torch.bool)
            nx = torch.full((self.vocab,), -1, dtype=torch.long)
            for sid in range(1, self.vocab):
                s = self.strs[sid]
                if not s:
                    continue
                n = grammar.walk(node, s)
                if n is not None:
                    m[sid] = True
                    nx[sid] = n
        self._cache[key] = (m, nx)
        return m.to(device), nx.to(device)


@torch.no_grad()
def constrained_generate(model, prompt_ids, grammar, table, block_size, device,
                         max_new=64, temperature=0.0, top_k=0):
    """Greedy (temperature=0) or sampled decoding under the grammar, batch of 1.

    Returns (generated_student_ids, text_pieces, stopped_ok).
    stopped_ok is True when the DFA reached an accepting state, i.e. the output
    is a complete, well-formed call by construction.
    """
    ids = list(prompt_ids)
    node = grammar.root
    out = []
    ok = False
    for _ in range(max_new):
        x = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        mask, nxt = table.mask(grammar, node, device=device)
        logits = logits.masked_fill(~mask, float("-inf"))
        if not torch.isfinite(logits).any():
            break                              # dead end (should not happen)
        if temperature and temperature > 0:
            if top_k:
                v, _ = torch.topk(logits, min(top_k, int(mask.sum())))
                logits = logits.masked_fill(logits < v[-1], float("-inf"))
            probs = torch.softmax(logits / temperature, dim=-1)
            tok = int(torch.multinomial(probs, 1))
        else:
            tok = int(logits.argmax())
        out.append(tok)
        ids.append(tok)
        node = int(nxt[tok])
        if grammar.accepting(node):
            ok = True
            break
        # In ANSWER mode allow the model to stop by emitting the EOS mark: the
        # FREE node has no EOS edge (every char loops), so check the text tail.
        if grammar.nodes[node]["kind"] == FREE:
            tail = "".join(table.strs[t] for t in out[-4:])
            if EOS_MARK in tail:
                ok = True
                break
    return out, "".join(table.strs[t] for t in out), ok
