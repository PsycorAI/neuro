"""Tool-calling SFT data for the flagship_v3 cognitive core (idea #4).

THE PROTOCOL (locked -- four places must agree byte for byte: this module,
scripts/gen_teacher_data.py, src/grammar.py and scripts/tool_eval.py):

    ### Tools:
    - calculator(expression: string): Evaluate an arithmetic expression.
    - get_weather(location: string, unit: celsius|fahrenheit): Get the current weather for a location.

    ### Instruction:
    What's 17% of 240?

    ### Response:
    TOOL {"name": "calculator", "arguments": {"expression": "0.17 * 240"}}

The response is ALWAYS a mode word followed by content: `TOOL <json>` for a
call, `ANSWER <text>` for a direct reply. Two reasons for the discriminator
rather than always-JSON:
  * free-text answers stay free text (no JSON string escaping for prose, which
    a 426M model would get wrong and which bloats the grammar), and
  * it gives grammar-constrained decoding a clean switch-on point: everything
    after `TOOL ` is constrained, `ANSWER ` is not.

Argument order is the order declared in tools.json, always. A single canonical
serialization is what lets the grammar be a small DFA instead of a JSON parser,
and it is what makes exact-match arg scoring meaningful.
"""
import hashlib
import json
import os

TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "training_data", "posttrain", "tools.json")

# Shared with sft_data.py -- the SFT stage's own end marker, encodable by our
# LUT vocab without special tokens.
EOS_MARK = "\n###"
MODE_TOOL = "TOOL"
MODE_ANSWER = "ANSWER"


def load_tools(path=TOOLS_PATH):
    with open(path) as f:
        return json.load(f)["tools"]


def tools_by_name(tools):
    return {t["name"]: t for t in tools}


def param_sig(p):
    if p["type"] == "enum":
        return f"{p['name']}: {'|'.join(p['values'])}"
    return f"{p['name']}: {p['type']}"


def render_tool_line(tool):
    sig = ", ".join(param_sig(p) for p in tool["params"])
    return f"- {tool['name']}({sig}): {tool['description']}"


def render_tools_block(offered, tools):
    """offered: list of tool names, rendered in the order given (the data
    generator shuffles them, so position carries no signal)."""
    by = tools_by_name(tools)
    lines = [render_tool_line(by[n]) for n in offered if n in by]
    return "### Tools:\n" + "\n".join(lines)


def dumps_call(name, arguments, tools):
    """Canonical serialization: args in schema order, `{"k": v, ...}` spacing.
    Unknown keys are dropped and missing keys omitted, so a malformed record
    can never silently teach a different shape than the grammar allows."""
    by = tools_by_name(tools)
    ordered = {}
    for p in by[name]["params"]:
        if p["name"] in arguments:
            v = arguments[p["name"]]
            ordered[p["name"]] = v
    return json.dumps({"name": name, "arguments": ordered},
                      separators=(", ", ": "), ensure_ascii=False)


def format_prompt(instruction, offered, tools):
    return (render_tools_block(offered, tools) + "\n\n### Instruction:\n"
            + instruction.strip() + "\n\n### Response:\n")


def say_of(name, tools):
    """Protocol v2 (2026-10-12, config tool_proto: 2): the fixed sentence the model writes before a call,
    `TOOL I will calculate that for you. {json}` (tools.json "say")."""
    return tools_by_name(tools)[name]["say"]


def full_offer(rec, tools, full_frac):
    """sft_tools_full_frac (2026-10-12, default 0 = unchanged): for that share of `tool` / `answer` records
    (hash-chosen) the offered list becomes all tools, shuffled deterministically -- training offers 2-6 tools, the demo
    all 9. Never for tool_absent (its gold tool must stay absent)."""
    if full_frac <= 0 or rec.get("kind") not in ("tool", "answer"):
        return rec["tools"]
    h = int(hashlib.md5(("full|" + rec["instruction"]).encode("utf-8")).hexdigest()[:8], 16)
    if (h % 1000) >= int(full_frac * 1000):
        return rec["tools"]
    names = [t["name"] for t in tools]
    return sorted(names, key=lambda n: hashlib.md5(f"{h}|{n}".encode()).hexdigest())


def format_response(rec, tools, proto=1):
    """rec is one line of training_data/posttrain/tools.jsonl."""
    if rec["target_mode"] == MODE_TOOL:
        call = rec["target_call"]
        say = f"{say_of(call['name'], tools)} " if proto >= 2 else ""
        return f"{MODE_TOOL} {say}{dumps_call(call['name'], call['arguments'], tools)}{EOS_MARK}"
    return f"{MODE_ANSWER} {rec['target_text'].strip()}{EOS_MARK}"


def sanitize_arg_values(rec, tools):
    """Strip characters the grammar deliberately cannot emit (a double quote or
    a backslash inside a string value) so the training target is always inside
    the constrained language. Without this the model would be trained on
    sequences the decoder can never produce."""
    if rec.get("target_mode") != MODE_TOOL:
        return rec
    args = rec["target_call"]["arguments"]
    for k, v in list(args.items()):
        if isinstance(v, str):
            args[k] = v.replace("\\", "").replace('"', "'")
    return rec


def split_of(instruction, test_frac=0.1):
    """Deterministic train/test assignment keyed on the instruction text.

    Hash-based rather than "last N lines": the JSONL is appended to by a
    resumable generator, so a positional split would silently reassign examples
    between runs and leak test data into training.
    """
    h = int(hashlib.md5(instruction.strip().encode("utf-8")).hexdigest()[:8], 16)
    return "test" if (h % 1000) < int(test_frac * 1000) else "train"


def load_tool_jsonl(path, tools, split=None, test_frac=0.1, proto=1, full_frac=0.0):
    """Read the generated JSONL into {prompt, response, kind, rec} records.
    split: None (all), "train" or "test". proto / full_frac: see format_response / full_offer."""
    out, bad = [], 0
    by = tools_by_name(tools)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1           # a partial last line from a killed writer
                continue
            try:
                if split and split_of(rec["instruction"], test_frac) != split:
                    continue
                if rec["target_mode"] == MODE_TOOL and rec["target_call"]["name"] not in by:
                    bad += 1; continue
                rec = sanitize_arg_values(rec, tools)
                rec["tools"] = full_offer(rec, tools, full_frac)
                out.append({
                    "prompt": format_prompt(rec["instruction"], rec["tools"], tools),
                    "response": format_response(rec, tools, proto),
                    "kind": rec.get("kind", "?"),
                    "rec": rec,
                })
            except (KeyError, TypeError):
                bad += 1
    return out, bad


def build_tool_examples(pairs, encode_fn, max_len):
    """Encode to (input_ids, labels) with loss masked to the response, matching
    sft_data.encode_with_mask so SFTLoader/collate work unchanged."""
    from sft_data import encode_with_mask
    out, skipped = [], 0
    for p in pairs:
        r = encode_with_mask(p["prompt"], p["response"], encode_fn, max_len)
        if r is None:
            skipped += 1
            continue
        out.append(r)
    return out, skipped


def parse_response(text):
    """Inverse of format_response, for eval. Returns
    (mode, call_dict_or_None, text_or_None). Tolerant of trailing EOS_MARK and
    of whitespace, strict about everything else -- an unparseable tool call must
    score as a failure, not get rescued."""
    t = text.strip()
    for cut in (EOS_MARK, "###"):
        if cut in t:
            t = t.split(cut)[0].strip()
    if t.startswith(MODE_TOOL):
        body = t[len(MODE_TOOL):].strip()
        if "{" in body:                      # protocol v2: skip the sentence before the call
            body = body[body.index("{"):]
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return MODE_TOOL, None, body
        if not isinstance(obj, dict) or "name" not in obj:
            return MODE_TOOL, None, body
        obj.setdefault("arguments", {})
        return MODE_TOOL, obj, body
    if t.startswith(MODE_ANSWER):
        return MODE_ANSWER, None, t[len(MODE_ANSWER):].strip()
    return None, None, t
