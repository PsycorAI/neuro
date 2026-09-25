"""Teacher-generated post-training data for flagship_v3 (ideas #1/#2/#4/#5).

One harness, four targets, one loaded teacher. Append-only JSONL plus a sidecar
progress file, so this can be killed at any moment (the user needs the 5080
back) and restarted with no duplicated and no lost work beyond the in-flight
batch.

  targets
    tools      -> TOOL/ANSWER function-calling SFT set WITH the two negative
                  classes that actually teach selection: irrelevant-tool (the
                  gold tool is absent from the offered list) and no-tool-needed
                  (a plain question). The 2026 tool-finetuning literature is
                  consistent that the negatives, not the positives, are what
                  move tool-selection accuracy.
    why        -> preference pairs whose CHOSEN response opens with one short
                  sentence of reasoning ("Teaching Claude Why": training on the
                  reasons moved misalignment 22%->3%, vs 22%->15% for training
                  on examples of correct behaviour alone).
    abstain    -> calibrated-abstention pairs in BOTH directions: unanswerable
                  (chosen = a specific IDK) and answerable (chosen = the answer,
                  rejected = a needless IDK). Both directions are required or
                  the model simply learns to always abstain.
    coldstart  -> teacher rollouts on instruction prompts: the off-policy cold
                  start that the on-policy-distillation recipe paper
                  (arXiv 2604.13016) found consistently improves final OPD
                  performance by closing the student/teacher style gap first.

  usage
    CUDA_VISIBLE_DEVICES=1 python scripts/gen_teacher_data.py \
        --target tools --n 24000 --batch 48 --max-minutes 90

GPU: always CUDA_VISIBLE_DEVICES=1 (the 5080). Device 0 is the live
flagship_v3 pretrain on the 5090 -- never touch it.
"""
import os, sys, json, time, random, argparse, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(ROOT, "training_data", "posttrain")
TOOLS_JSON = os.path.join(OUT_DIR, "tools.json")

# Teacher preference order. The OPD teacher must be a Llama-3-tokenizer model:
# our 32768 student vocab is a LUT projection of Llama-3's 128256, so token
# boundaries have to match for per-token reverse KL to line up at all. Phi-3.5
# (already cached) is a text-only fallback -- usable for these datasets, NOT
# usable as an OPD scoring teacher later.
TEACHERS = ["meta-llama/Llama-3.2-3B-Instruct",
            "unsloth/Llama-3.2-3B-Instruct",
            "microsoft/Phi-3.5-mini-instruct"]


# ---------------------------------------------------------------- teacher ---
def _weights_complete(snap_dir):
    """True only if every weight shard is actually on disk. Checking config.json
    alone is not enough: config.json is the FIRST file a snapshot_download
    fetches, so a half-finished download would resolve and then blow up in
    from_pretrained."""
    import glob as _glob
    shards = _glob.glob(os.path.join(snap_dir, "*.safetensors"))
    if not shards:
        return False
    index = os.path.join(snap_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            want = set(json.load(f)["weight_map"].values())
        have = {os.path.basename(s) for s in shards}
        return want.issubset(have)
    return True


def _is_llama3_tokenizer(repo):
    from transformers import AutoTokenizer
    try:
        return len(AutoTokenizer.from_pretrained(repo)) > 100000
    except Exception:
        return False


def resolve_teacher(preferred=None, require_llama=False):
    """require_llama: only accept a ~128k-vocab (Llama-3) tokenizer. Used for the
    coldstart target, whose whole purpose is to pre-align the student with the
    teacher that will later score tokens during OPD -- a different tokenizer
    there would defeat the stage. Text-only targets accept any teacher."""
    from huggingface_hub import snapshot_download
    cands = ([preferred] if preferred else []) + TEACHERS
    for repo in cands:
        try:
            snap = snapshot_download(repo, local_files_only=True)
        except Exception:
            continue
        if not _weights_complete(snap):
            print(f"  [skip] {repo}: weights incomplete on disk", flush=True)
            continue
        if require_llama and not _is_llama3_tokenizer(repo):
            print(f"  [skip] {repo}: not a Llama-3 tokenizer (--require-llama)",
                  flush=True)
            continue
        return repo
    raise RuntimeError(f"no usable teacher available locally; tried {cands} "
                       f"(require_llama={require_llama})")


def load_teacher(model_id, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    # Decoder-only batched generation needs LEFT padding, or short prompts get
    # their continuation glued on after the pad run and come back as garbage.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def batched_chat(model, tok, indexed_prompts, max_new, temperature, batch_size,
                 system=None, deadline=None, min_batch=4):
    """indexed_prompts: list of (item_index, prompt_text).
    Yields (batch_end_index, [(item_index, generated_text), ...]) per batch so
    the caller can flush and record progress at a batch boundary.

    The batch size self-tunes DOWN on OOM and never back up. KV-cache cost per
    token varies ~3.4x between teachers (Llama-3.2-3B uses GQA with 8 KV heads;
    Phi-3.5-mini has no GQA and costs ~393KB/token), and prompt lengths vary per
    target, so a single hard-coded batch either wastes the 5080 or dies. Start
    high, back off, keep going -- the point is maximum throughput without
    babysitting.
    """
    import torch
    bs = batch_size
    i = 0
    n = len(indexed_prompts)
    while i < n:
        if deadline and time.time() > deadline:
            print("  [budget] time budget reached, stopping cleanly", flush=True)
            return
        chunk = indexed_prompts[i:i + bs]
        texts = []
        for _, p in chunk:
            msgs = ([{"role": "system", "content": system}] if system else [])
            msgs.append({"role": "user", "content": p})
            texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True))
        try:
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new,
                                     do_sample=True, temperature=temperature,
                                     top_p=0.95, pad_token_id=tok.pad_token_id)
        except Exception as e:
            # Memory pressure on this card does not always say "out of memory":
            # at batch 48 with Phi-3.5-mini it surfaced as
            # CUBLAS_STATUS_INTERNAL_ERROR (cuBLAS failing to get its
            # workspace). Treat the allocator-adjacent CUDA failures as OOM.
            msg = str(e).lower()
            oom = (isinstance(e, torch.cuda.OutOfMemoryError)
                   or "out of memory" in msg
                   or "cublas_status_internal_error" in msg
                   or "cublas_status_alloc_failed" in msg)
            if not oom or bs <= min_batch:
                raise
            torch.cuda.empty_cache()
            bs = max(min_batch, bs // 2)
            print(f"  [oom] batch -> {bs}, retrying", flush=True)
            continue
        gen = out[:, enc["input_ids"].shape[1]:]
        res = [(chunk[j][0], tok.decode(g, skip_special_tokens=True).strip())
               for j, g in enumerate(gen)]
        i += len(chunk)
        yield chunk[-1][0], res


# ------------------------------------------------------------ tool target ---
CITIES = ["Paris", "Tokyo", "Lagos", "Sao Paulo", "Mumbai", "Reykjavik", "Seattle",
          "Cairo", "Toronto", "Berlin", "Seoul", "Auckland", "Nairobi", "Lima",
          "Dublin", "Helsinki", "Bogota", "Hanoi", "Perth", "Marrakesh"]
TIMEZONES = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo", "Australia/Sydney",
             "America/Los_Angeles", "Europe/Berlin", "Asia/Kolkata", "America/Sao_Paulo",
             "Africa/Cairo", "Pacific/Auckland", "Asia/Shanghai"]
UNITS = [("miles", "kilometers"), ("celsius", "fahrenheit"), ("pounds", "kilograms"),
         ("feet", "meters"), ("gallons", "liters"), ("ounces", "grams"),
         ("inches", "centimeters"), ("acres", "hectares"), ("knots", "mph")]
LANGS = ["Spanish", "French", "Japanese", "German", "Portuguese", "Arabic",
         "Hindi", "Korean", "Italian", "Swahili", "Dutch", "Polish"]
TERMS = ["photosynthesis", "entropy", "sonata", "arbitrage", "mitosis", "quorum",
         "isotope", "allegory", "cadence", "escrow", "tundra", "syntax",
         "inflation", "catalyst", "diaspora", "latency", "osmosis", "parallax"]
SEARCH_TOPICS = ["the current population of {c}", "who won the last election in {c}",
                 "today's exchange rate for the euro", "the latest SpaceX launch date",
                 "recent news about renewable energy", "the current price of copper",
                 "this year's Nobel Prize in physics", "the opening hours of the Louvre",
                 "the tallest building completed this year", "current flight delays at Heathrow"]
CONTACTS = ["Maya", "Dad", "Priya", "the team", "Sam", "my landlord", "Dr. Okafor",
            "Jordan", "the group chat", "Ines"]
REMINDERS = [("call the dentist", "tomorrow at 9am"), ("pay the electricity bill", "Friday 6pm"),
             ("water the plants", "Sunday morning"), ("submit the timesheet", "Thursday at 5pm"),
             ("pick up the parcel", "Saturday at noon"), ("renew the passport", "next Monday 10am"),
             ("back up the laptop", "tonight at 11pm"), ("book the train", "tomorrow evening")]
PHRASES = ["where is the train station", "thank you for your help",
           "I would like a table for two", "how much does this cost",
           "my flight is delayed", "can you speak more slowly",
           "happy birthday", "I am allergic to peanuts"]
MSG_BODIES = ["running about 10 minutes late", "the meeting moved to 3pm",
              "dinner is at the usual place", "I got the job",
              "can you grab milk on the way home", "the package arrived"]

REFUSALS = [
    "I can't do that with the tools I have here.",
    "I don't have a tool for that one.",
    "That's outside what these tools can do.",
    "None of the available tools cover that.",
    "I'd need a tool I don't have access to here.",
    "There's no tool available for that request.",
    "I can't look that up with the tools I've got.",
    "That isn't something these tools can handle.",
    "I don't have the right tool for that.",
    "No tool here can do that, sorry.",
    "That one's beyond the tools available to me.",
    "I'm not able to do that with the current toolset.",
]


def sample_call(tool, rng):
    """Return (arguments dict, a short natural hint for the query writer)."""
    n = tool["name"]
    if n == "search":
        t = rng.choice(SEARCH_TOPICS).replace("{c}", rng.choice(CITIES))
        return {"query": t}, f"wants to look up {t}"
    if n == "calculator":
        a, b = rng.randint(2, 999), rng.randint(2, 99)
        expr = rng.choice([f"{a} * {b}", f"{a} / {b}", f"{a} + {b} * 2",
                           f"{a} - {b}", f"0.{rng.randint(5, 95):02d} * {a}",
                           f"({a} + {b}) / 3", f"{a} ** 2"])
        return {"expression": expr}, f"wants the value of {expr}"
    if n == "unit_convert":
        f, t = rng.choice(UNITS)
        v = round(rng.uniform(1, 500), rng.choice([0, 1]))
        return {"value": v, "from_unit": f, "to_unit": t}, f"wants {v} {f} in {t}"
    if n == "get_weather":
        c, u = rng.choice(CITIES), rng.choice(["celsius", "fahrenheit"])
        return {"location": c, "unit": u}, f"wants the weather in {c} in {u}"
    if n == "get_time":
        z = rng.choice(TIMEZONES)
        return {"timezone": z}, f"wants the current time in {z}"
    if n == "define":
        t = rng.choice(TERMS)
        return {"term": t}, f"wants the definition of '{t}'"
    if n == "translate":
        p, lg = rng.choice(PHRASES), rng.choice(LANGS)
        return {"text": p, "target_lang": lg}, f"wants '{p}' translated into {lg}"
    if n == "set_reminder":
        txt, when = rng.choice(REMINDERS)
        return {"text": txt, "datetime": when}, f"wants a reminder to {txt} {when}"
    if n == "send_message":
        who, body = rng.choice(CONTACTS), rng.choice(MSG_BODIES)
        return {"recipient": who, "body": body}, f"wants to message {who} saying '{body}'"
    raise ValueError(n)


QUERY_SYS = ("You write short, natural user requests for a voice assistant. "
             "Output ONLY a numbered list, one request per line, nothing else.")


def build_tool_items(n_items, seed, tools):
    rng = random.Random(seed)
    items = []
    for _ in range(n_items):
        tool = rng.choice(tools)
        args, hint = sample_call(tool, rng)
        items.append({"tool": tool["name"], "arguments": args, "prompt": (
            f"A user {hint}. Write 4 different ways a real person might ask for "
            f"exactly that, in one sentence each. Vary the phrasing, length and "
            f"formality. Do not mention tools or JSON.")})
    return items


def parse_numbered(text, want=4):
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+)$", line.strip())
        if m:
            q = m.group(1).strip().strip('"').strip()
            if 3 <= len(q.split()) <= 40:
                out.append(q)
    return out[:want]


def free_items_tools(n, seed, tools):
    """No-tool-needed negatives from Alpaca. No teacher call: the reference
    answer is already in the dataset."""
    from datasets import load_dataset
    rng = random.Random(seed + 7)
    names = [t["name"] for t in tools]
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    out, k = [], 0
    for ex in ds:
        inp = (ex.get("input") or "").strip()
        q = ex["instruction"].strip() + (("\n\n" + inp) if inp else "")
        a = ex["output"].strip()
        if not (20 < len(q) < 400 and 10 < len(a) < 600):
            continue
        k += 1
        if k % 3:                       # thin out for topical variety
            continue
        out.append({"kind": "answer", "instruction": q,
                    "tools": rng.sample(names, k=rng.randint(3, 6)),
                    "target_mode": "ANSWER", "target_text": a})
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------- why target ---
WHY_SYS = ("You rewrite assistant replies so the reasoning behind them is visible. "
           "Output ONLY the rewritten reply.")


def build_why_items(n_items, seed):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    items = []
    for i in idx:
        ex = ds[i]
        try:
            prompt = (ex.get("prompt") or ex["chosen"][0]["content"]).strip()
            chosen = ex["chosen"][-1]["content"].strip()
            rejected = ex["rejected"][-1]["content"].strip()
        except (KeyError, IndexError, TypeError):
            continue
        if not (20 < len(prompt) < 600 and 40 < len(chosen) < 1200 and rejected):
            continue
        items.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "gen_prompt": (
                          "Here is a user request and a good assistant reply.\n\n"
                          f"USER: {prompt}\n\nREPLY: {chosen}\n\n"
                          "Rewrite the reply so it OPENS with one short sentence "
                          "stating the reason behind the approach it takes -- why "
                          "this is the right way to answer, or why a caveat "
                          "matters. Keep the rest of the substance the same. Do "
                          "not add headings or meta-commentary.")})
        if len(items) >= n_items:
            break
    return items


# --------------------------------------------------------- abstain target ---
IDK_TEMPLATES = [
    "I don't know {t}. That's a specific fact I'd need to look up rather than guess at.",
    "I'm not able to answer that reliably -- {t} isn't something I can recall with confidence.",
    "I don't have a dependable answer for {t}. I'd rather say so than invent a number.",
    "That one I genuinely don't know. {T} would need a lookup to answer properly.",
    "I can't answer {t} accurately. Guessing here would be worse than admitting the gap.",
    "I don't know. {T} is the kind of detail I'd need to verify before stating it.",
]
NEEDLESS_IDK = ["I'm not sure about that one.", "I don't know, sorry.",
                "That's outside what I can answer.", "I couldn't say.",
                "I have no way of knowing that."]
OBSCURE_TOPICS = ["a minor 19th-century civil engineer", "a regional sports record",
                  "a discontinued consumer product's serial numbering",
                  "the attendance at a specific provincial match",
                  "a small municipality's budget line",
                  "an obscure academic's dissertation title",
                  "a minor character in a 1970s television series",
                  "the registration number of a decommissioned ferry",
                  "a local road's renaming date", "a defunct company's last board member",
                  "a specific issue of a small-press comic",
                  "a village's population in a specific census"]
OBSCURE_SYS = ("You invent obscure trivia questions for a calibration test. "
               "Follow the output format exactly.")


def build_abstain_items(n_items, seed):
    rng = random.Random(seed)
    return [{"gen_prompt": (
        f"Invent one very obscure factual question about {rng.choice(OBSCURE_TOPICS)}. "
        f"The answer must be a specific name, number or date that almost nobody "
        f"would know offhand.\nThen invent a plausible-sounding but fabricated "
        f"specific answer.\nOutput exactly two lines:\nQ: <question>\n"
        f"A: <fabricated specific answer>")} for _ in range(n_items)]


def free_items_abstain(n, seed):
    """Answerable direction, free from SciQ (question + correct answer).
    chosen = the real answer, rejected = a needless abstention."""
    from datasets import load_dataset
    # namespaced id: the bare "sciq" alias no longer resolves under
    # huggingface_hub 1.x (Invalid HF URI -- needs 'namespace/name').
    ds = load_dataset("allenai/sciq", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        q, a = ex["question"].strip(), ex["correct_answer"].strip()
        if not q or not a or len(q) < 15:
            continue
        out.append({"prompt": q, "chosen": a, "rejected": rng.choice(NEEDLESS_IDK),
                    "pair_type": "answerable"})
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------- coldstart target ---
def build_coldstart_items(n_items, seed):
    from datasets import load_dataset
    rng = random.Random(seed)
    prompts = []
    for ex in load_dataset("yahma/alpaca-cleaned", split="train"):
        inp = (ex.get("input") or "").strip()
        p = ex["instruction"].strip() + (("\n\n" + inp) if inp else "")
        if 20 < len(p) < 700:
            prompts.append(p)
    rng.shuffle(prompts)
    return [{"prompt": p, "gen_prompt": p} for p in prompts[:n_items]]


# ------------------------------------------------------------- bookkeeping --
def count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_progress(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"free": -1, "teach": -1}


def save_progress(path, prog):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["tools", "why", "abstain", "coldstart"])
    ap.add_argument("--n", type=int, default=20000, help="target line count")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-llama", action="store_true",
                    help="only accept a Llama-3-tokenizer teacher (coldstart)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.target}.jsonl")
    prog_path = os.path.join(OUT_DIR, f"{args.target}.progress.json")
    done_path = out_path + ".done"
    if os.path.exists(done_path):
        print(f"{args.target}: .done present -- nothing to do"); return

    written = count_lines(out_path)
    prog = load_progress(prog_path)
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    print(f"{args.target}: resume at {written} lines "
          f"(free={prog['free']}, teach={prog['teach']}) -> target {args.n}", flush=True)
    if written >= args.n:
        open(done_path, "w").write(str(written)); print("already at target"); return

    with open(TOOLS_JSON) as f:
        tools = json.load(f)["tools"]
    t0 = time.time()
    fout = open(out_path, "a", encoding="utf-8")

    def emit(rec):
        nonlocal written
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1

    # ---- phase 1: records that need no teacher call (write these first) -----
    free_frac = {"tools": 0.25, "abstain": 0.40}.get(args.target, 0.0)
    if free_frac:
        n_free = int(args.n * free_frac)
        builder = free_items_tools if args.target == "tools" else free_items_abstain
        need = n_free - (prog["free"] + 1)
        if need > 0:
            print(f"  phase 1: {need} teacher-free records...", flush=True)
            items = (builder(n_free, args.seed, tools) if args.target == "tools"
                     else builder(n_free, args.seed))
            for i in range(prog["free"] + 1, min(n_free, len(items))):
                emit(items[i]); prog["free"] = i
            fout.flush(); save_progress(prog_path, prog)
            print(f"  -> {written} lines", flush=True)

    if written >= args.n:
        fout.close(); open(done_path, "w").write(str(written))
        print(f"done (teacher not needed): {written} lines"); return

    # ---- phase 2: teacher generation ---------------------------------------
    repo = resolve_teacher(args.teacher, require_llama=args.require_llama)
    print(f"teacher: {repo}", flush=True)
    model, tok = load_teacher(repo, args.device)
    rng = random.Random(args.seed + 99)
    start = prog["teach"] + 1

    if args.target == "tools":
        items = build_tool_items(start + args.n, args.seed, tools)
        prompts = [(i, items[i]["prompt"]) for i in range(start, len(items))]
        stream = batched_chat(model, tok, prompts, 200, args.temperature,
                              args.batch, system=QUERY_SYS, deadline=deadline)
        for last_i, res in stream:
            for i, text in res:
                it = items[i]
                for q in parse_numbered(text):
                    if written >= args.n:
                        break
                    others = [t["name"] for t in tools if t["name"] != it["tool"]]
                    distract = rng.sample(others, k=rng.randint(2, 5))
                    offered = distract + [it["tool"]]
                    rng.shuffle(offered)
                    emit({"kind": "tool", "instruction": q, "tools": offered,
                          "target_mode": "TOOL",
                          "target_call": {"name": it["tool"],
                                          "arguments": it["arguments"]}})
                    # irrelevant-tool negative: same query, gold tool REMOVED
                    if rng.random() < 0.25 and written < args.n:
                        emit({"kind": "tool_absent", "instruction": q,
                              "tools": sorted(distract), "target_mode": "ANSWER",
                              "target_text": rng.choice(REFUSALS)})
            prog["teach"] = last_i
            fout.flush(); save_progress(prog_path, prog)
            print(f"  {written}/{args.n} | {(time.time()-t0)/60:.1f} min", flush=True)
            if written >= args.n:
                break

    elif args.target == "why":
        items = build_why_items(start + args.n, args.seed)
        prompts = [(i, items[i]["gen_prompt"]) for i in range(start, len(items))]
        stream = batched_chat(model, tok, prompts, 400, 0.7, args.batch,
                              system=WHY_SYS, deadline=deadline)
        for last_i, res in stream:
            for i, text in res:
                it = items[i]
                if not (40 < len(text) < 2000) or written >= args.n:
                    continue
                # (a) reasons AND quality: why-narrated chosen vs original rejected
                emit({"prompt": it["prompt"], "chosen": text,
                      "rejected": it["rejected"], "pair_type": "why_vs_rejected"})
                # (b) reasons ONLY: same answer, narrated vs unnarrated. This is
                # the pair that isolates "state the reason" from "be correct".
                if written < args.n:
                    emit({"prompt": it["prompt"], "chosen": text,
                          "rejected": it["chosen"], "pair_type": "why_vs_bare"})
            prog["teach"] = last_i
            fout.flush(); save_progress(prog_path, prog)
            print(f"  {written}/{args.n} | {(time.time()-t0)/60:.1f} min", flush=True)
            if written >= args.n:
                break

    elif args.target == "abstain":
        items = build_abstain_items(start + args.n, args.seed)
        prompts = [(i, items[i]["gen_prompt"]) for i in range(start, len(items))]
        stream = batched_chat(model, tok, prompts, 120, 1.0, args.batch,
                              system=OBSCURE_SYS, deadline=deadline)
        for last_i, res in stream:
            for i, text in res:
                q = a = None
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("Q:"):
                        q = line[2:].strip()
                    elif line.startswith("A:"):
                        a = line[2:].strip()
                if not q or not a or len(q) < 20 or len(a) < 2 or written >= args.n:
                    continue
                topic = q.rstrip("?").lower()
                if len(topic) > 90:
                    topic = topic[:90].rsplit(" ", 1)[0]
                idk = rng.choice(IDK_TEMPLATES).format(
                    t=topic, T=topic[:1].upper() + topic[1:])
                emit({"prompt": q, "chosen": idk, "rejected": a,
                      "pair_type": "unanswerable"})
            prog["teach"] = last_i
            fout.flush(); save_progress(prog_path, prog)
            print(f"  {written}/{args.n} | {(time.time()-t0)/60:.1f} min", flush=True)
            if written >= args.n:
                break

    elif args.target == "coldstart":
        items = build_coldstart_items(start + args.n + 2000, args.seed)
        prompts = [(i, items[i]["gen_prompt"]) for i in range(start, len(items))]
        # 256 not 320: alpaca-style answers rarely need more, and every extra
        # generated token costs KV cache on a card that is already close to the
        # WDDM spill point at this batch size.
        stream = batched_chat(model, tok, prompts, 256, args.temperature,
                              args.batch, deadline=deadline)
        for last_i, res in stream:
            for i, text in res:
                if len(text) >= 20 and written < args.n:
                    emit({"instruction": items[i]["prompt"], "output": text})
            prog["teach"] = last_i
            fout.flush(); save_progress(prog_path, prog)
            print(f"  {written}/{args.n} | {(time.time()-t0)/60:.1f} min", flush=True)
            if written >= args.n:
                break

    fout.close()
    el = (time.time() - t0) / 60
    if written >= args.n:
        open(done_path, "w").write(str(written))
        print(f"DONE {args.target}: {written} lines in {el:.1f} min")
    else:
        print(f"PARTIAL {args.target}: {written}/{args.n} in {el:.1f} min (rerun to resume)")


if __name__ == "__main__":
    main()
