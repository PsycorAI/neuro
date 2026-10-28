"""demo/guards.py -- 2026-10-18 (Neuro 68). Backend guards around the 1B model's tool calls, no torch (unit-testable:
temp/n68_test_guards.py). Two entry points, both used by engine.Engine.assistant and both reported to the page, so the
demo stays honest about what the model did and what the backend fixed:

  ground(call, question) -> str | None   repairs arguments the model mis-copied from the question (asked "London",
                                         called "Lancaster"; "call my son" saved as "call the doctor"). Only touches a
                                         value that is NOT in the question when the question clearly names another one.
                                         Returns a short note of the model's original values, or None.
  route(question, say) -> call | None    an explicit request ("search for X", "what's the weather in X", "remind me to
                                         X") that the model answered from memory instead of calling the tool."""
import re

_PLACE = r"([A-Z][\w'.-]*(?:\s+(?:[A-Z][\w'.-]*|de|del|la|le|of|on|upon))*(?:\s+[A-Z][\w'.-]*)?)"
_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_LANGS = ("spanish", "french", "german", "italian", "portuguese")


def _in(v, q):
    return str(v).strip().lower() in q.lower() if str(v).strip() else False


def _nums(s):
    return [float(x.replace(",", "")) for x in re.findall(_NUM, str(s))]


_STOP = {"the", "what", "who", "whom", "when", "where", "which", "why", "how", "was", "were", "are", "is", "did", "does",
         "and", "for", "that", "this", "with", "from", "about", "tell", "can", "could", "would", "you", "please", "me",
         "find", "out", "search", "look", "know", "give", "some", "any", "latest", "news", "info", "information",
         "thanks", "thank", "hello", "hey", "bye", "goodbye", "okay", "great", "cool", "nice", "good", "morning",
         "evening", "night", "yes", "yeah", "sure", "fine", "sorry"}


def _topic(q):
    """"Can you tell me about the Eiffel Tower?" -> "the Eiffel Tower"."""
    t = re.sub(r"^\s*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:tell\s+me\s+(?:about|who|what)|"
               r"what\s+do\s+you\s+know\s+about|search\s+for|look\s+up|find\s+out\s+about)\s+", "", q.strip(), flags=re.I)
    return t.rstrip(" ?.!") or q.strip()


def _place(q):
    """The place a question names: "in/for/at <Place>" (capitalised words)."""
    m = re.search(r"\b(?:in|for|at)\s+" + _PLACE, q)
    return re.split(r"[.?!,]\s", m.group(1))[0].rstrip(".") if m else None   # 10-21: not into the next sentence


# 10-21 (Glenn): time / weather questions that name no place mean HERE (tools_exec.HOME). Whole-message patterns only
# ("What time does the NYSE open?", "What's the weather on the east coast?" are not matched)
_HERE = r"(?:\s+(?:like|now|right\s+now|here|today|tonight|tomorrow|outside|out\s+there|currently|please|this\s+(?:morning|afternoon|evening)))*\s*[?.!]*\s*$"
_LEAD = r"^\s*(?:(?:hey|hi|ok|okay|so)[\s,]+)?(?:(?:can|could)\s+you\s+tell\s+me\s+|do\s+you\s+know\s+|please\s+)?"
_LOCAL_TIME = re.compile(_LEAD + r"(?:what(?:'s|\s+is)\s+the\s+(?:current\s+)?(?:time|date)|what\s+time\s+(?:is\s+it|it\s+is)|"
                         r"what\s+(?:day|date)\s+(?:is\s+it|it\s+is)|what(?:'s|\s+is)\s+today's\s+date|the\s+time)" + _HERE, re.I)
_LOCAL_WEATHER = re.compile(_LEAD + r"(?:what(?:'s|\s+is)\s+the\s+(?:weather|temperature|forecast)|how(?:'s|\s+is)\s+the\s+weather|"
                            r"(?:is\s+it|will\s+it)\s+(?:going\s+to\s+)?(?:rain|snow)(?:ing)?|is\s+it\s+(?:cold|hot|warm|sunny)|"
                            r"do\s+i\s+need\s+an?\s+(?:umbrella|jacket|coat)|the\s+weather)" + _HERE, re.I)


def _name(q):
    """10-21 (chat brain): a follow-up's place has no "in/for/at" ("What about London?", "Sorry, I meant Rome."):
    the first capitalised name that does not start a sentence."""
    m = re.search(r"(?<!^)(?<![.?!]\s)\b([A-Z][a-zA-Z'.-]{2,}(?:\s+[A-Z][a-zA-Z'.-]+)*)", q.strip())
    return m.group(1).rstrip(".") if m else None


def local_time(q):
    return _LOCAL_TIME.match(q.strip()) is not None


def local_weather(q):
    return _LOCAL_WEATHER.match(q.strip()) is not None


def _unit(q):
    ql = q.lower()
    if re.search(r"\b(celsius|centigrade|°c)\b", ql):
        return "celsius"
    if re.search(r"\b(fahrenheit|°f)\b", ql):
        return "fahrenheit"
    return None


def _reminder(q):
    """"remind me to <task> [<when>]" -> (task, when or None)."""
    m = re.search(r"\bremind me to\s+(.+?)[.?!]?\s*$", q.strip(), re.I)
    if not m:
        return None
    body = m.group(1)
    day = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d"
    w = re.search(r"\s+((?:(?:at|by|before|after)\s+(?:\d|noon|midnight|tonight|tomorrow)|on\s+(?:" + day + r")|in\s+(?:\d|an?\s|half)"
                  r"|tomorrow|tonight|today|this\s+(?:morning|afternoon|evening|week)|next\s+\w+).*)$", body, re.I)
    task, when = (body[:w.start()], w.group(1)) if w else (body, None)
    return re.sub(r"\bmy\b", "your", task, flags=re.I).strip(), when


def _arith(q):
    """The arithmetic a question asks for, as a calculator expression, or None."""
    e = re.sub(r"^\s*(?:what\s+is|what's|calculate|compute|how\s+much\s+is|evaluate)\s+", "", q.strip(), flags=re.I)
    e = e.rstrip(" ?.!")
    for a, b in ((r"\btimes\b|\bmultiplied by\b|\bx\b", "*"), (r"\bplus\b", "+"), (r"\bminus\b", "-"),
                 (r"\bdivided by\b|\bover\b", "/"), (r"\bsquared\b", "**2"), (r"\bto the power of\b", "**")):
        e = re.sub(a, b, e, flags=re.I)
    e = re.sub(r"^the\s+", "", e, flags=re.I)
    e = re.sub(r"\bsquare root of\s+(" + _NUM + ")", r"sqrt(\1)", e, flags=re.I).strip()
    if not re.fullmatch(r"(?:[\d\s.,+\-*/^()%]|\bof\b|sqrt)+", e, re.I):
        return None
    return e if re.search(r"\d", e) and re.search(r"[+\-*/^%]|sqrt", e) else None


def ground(call, question, offline=False, context=""):
    import tools_exec
    a = call.get("arguments")
    if not isinstance(a, dict):
        return None
    for k, v in a.items():                      # " Anna" -> "Anna" (whitespace only: not reported as a repair)
        if isinstance(v, str):
            a[k] = v.strip()
    q, name, old, home = question, call.get("name"), dict(a), set()
    if name == "get_weather":
        p, loc = _place(q) or (_name(q) if context else None), str(a.get("location", ""))
        # not in the question, or a part of the place it names ("Stratford" for "Stratford upon Avon")
        if p and (not _in(loc, q) or (loc.lower() in p.lower() and len(p) > len(loc))):
            a["location"] = p
        elif not p and local_weather(q) and not _in(loc, q) and loc != tools_exec.HOME["city"]:
            a["location"] = tools_exec.HOME["city"]; home.add("location")     # 10-21: no place named = here
        u = _unit(q) or tools_exec.home_unit()                                # 10-21: else the home units
        if a.get("unit") != u:
            a["unit"] = u
            if not _unit(q):
                home.add("unit")
    elif name == "get_time":
        p = _place(q) or (_name(q) if context else None)
        if p and not _in(a.get("timezone", ""), q):
            mine, theirs = tools_exec.get_time(p, offline), tools_exec.get_time(a.get("timezone", ""), offline)
            if mine["ok"] and mine["text"].rsplit("(", 1)[-1] != theirs["text"].rsplit("(", 1)[-1]:
                a["timezone"] = p
        elif not p and local_time(q) and a.get("timezone") != tools_exec.HOME["tz"]:
            a["timezone"] = tools_exec.HOME["tz"]; home.add("timezone")       # 10-21: no place named = here
    elif name == "set_reminder":
        r = _reminder(q)
        if r:
            if not _in(a.get("text", ""), q):
                a["text"] = r[0]
            norm = lambda s: re.sub(r"\b(?:at|on)\b|[\s.]", "", str(s).lower())
            if r[1] and norm(a.get("datetime", "")) != norm(r[1]):
                a["datetime"] = r[1]
    elif name == "calculator":
        # the question is itself arithmetic and the model's expression computes something else ("square root of 144"
        # called as "144 ** 2"; "17% of 240" as "17% of 204") -> the question's own expression
        e = _arith(q)
        if e:
            mine, theirs = tools_exec.calculator(e), tools_exec.calculator(a.get("expression", ""))
            if mine["ok"] and mine["text"] != theirs["text"]:
                a["expression"] = e
    elif name == "search":
        # a query that shares no content word with the question ("tell me about the Eiffel Tower" searched as "the
        # tallest building completed this year") -> the question's own topic
        cw = lambda t: {w for w in re.findall(r"[a-z0-9]{3,}", str(t).lower()) if w not in _STOP}
        # 10-20: or one that drops a name the question gives ("the Treaty of Versaille" searched as "the treaty that
        # ended World War I" -> the World War I article, no treaty date; Wikipedia's search tolerates the misspelling)
        # (a name the model spelled differently, "Einstein" for "Einstien", counts as kept)
        import difflib
        names = {w.lower() for w in re.findall(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z'-]{3,}", q.strip())} - _STOP
        dropped = [w for w in names if not difflib.get_close_matches(w, cw(a.get("query", "")), 1, 0.8)]
        # 10-21 (chat brain): a follow-up's query may come from the earlier turns ("When was she born?" ->
        # "Jane Austen birth date"): kept when it shares a content word with them (`context` = earlier user lines)
        if cw(q) and (not (cw(q) & cw(a.get("query", ""))) or dropped) and not (cw(context) & cw(a.get("query", ""))):
            a["query"] = _topic(q)
    elif name == "unit_convert":
        qn = _nums(q)
        try:
            v = float(a.get("value"))
        except (TypeError, ValueError):
            v = None
        if len(qn) == 1 and v != qn[0]:
            a["value"] = qn[0]
    elif name == "define":
        m = re.search(r"\b(?:define|definition of|meaning of|what does)\s+(?:the\s+word\s+)?['\"‘’“”]?([A-Za-z][\w-]*(?:\s[\w-]+)?)['\"‘’“”]?"
                      r"(?=\s+mean|\s*[?.!]?\s*$)", q, re.I)
        if m and not _in(a.get("term", ""), q):
            a["term"] = m.group(1)
    elif name == "translate":
        m = re.search(r"['\"“‘](.+?)['\"”’]", q) or re.search(r"\btranslate\s+(.+?)\s+(?:in)?to\s+\w+", q, re.I)
        if m and not _in(a.get("text", ""), q):
            a["text"] = m.group(1)
        lang = re.search(r"\b(?:in|into|to)\s+(" + "|".join(_LANGS) + r")\b", q, re.I)
        if lang and str(a.get("target_lang", "")).lower() != lang.group(1).lower():
            a["target_lang"] = lang.group(1).capitalize()
    elif name == "send_message":
        m = re.search(r"\b(?:message|text|tell|write to)\s+(?:to\s+)?(my\s+\w+|[A-Z]\w+)", q)
        if m and not _in(a.get("recipient", ""), q):
            a["recipient"] = re.sub(r"^my\s+", "", m.group(1), flags=re.I)
        b = re.search(r"\b(?:saying|that says|that|:)\s+(.+?)[.!]?\s*$", q, re.I)
        # 10-21: "Send that to Anna too." names no body ("that" = the earlier message), the regex would take "to Anna too"
        if b and not re.match(r"to\b", b.group(1), re.I) and not _in(a.get("body", ""), q):
            a["body"] = b.group(1)
    if a == old:
        return None
    return ", ".join(f"{k}={old.get(k)!r}" + (" (set to this computer's location / units)" if k in home else "")
                     for k in a if a.get(k) != old.get(k))


_TRIGGER = {"get_weather": r"weather|temperature|forecast|rain|snow|sunny|hot|cold|wind|umbrella|jacket|coat",
            "get_time": r"\btime\b|clock|what day|date", "calculator": r"\d", "unit_convert": r"\d",
            # 10-20: a message only when one is asked for ("Order a pizza" sent "I got the pizza" to "the team")
            "send_message": r"\bmessage|\btext\b|\btell\b|\be-?mail|\bwrite\b|\bsend\b|\bnotify|\blet\s+\w+\s+know"}


def groundless(call, question, context=""):
    """True when the model's tool call has nothing to do with the message ("hi" -> translate "can you speak more
    slowly" into Arabic): no content word of the message in the call's arguments, and for the tools that need no
    words from the message (weather, time, numbers) none of their trigger words either. Checked before ground()."""
    name, a = call.get("name"), call.get("arguments")
    if not isinstance(a, dict):
        return False
    cw = lambda t: {w for w in re.findall(r"[a-z0-9]{3,}", str(t).lower()) if w not in _STOP}
    c = {"name": name, "arguments": dict(a)}
    if name in _TRIGGER:
        if re.search(_TRIGGER[name], question, re.I):
            return False
        # 10-21 (chat brain): a follow-up ("What about London?", "And in Berlin?") gets the trigger from the earlier
        # turns, but the repaired call must still share a word with the message ("What is my name?" after a weather
        # turn -> get_weather Paris stays groundless)
        if not (context and re.search(_TRIGGER[name], context, re.I)):
            return True
        ground(c, question, context=context)
        return not (cw(question) & cw(" ".join(str(v) for v in c["arguments"].values())))
    ground(c, question, context=context)        # what the backend would repair is not groundless
    cwn = lambda t: cw(t) | set(re.findall(r"\d+", str(t)))   # 10-21: + numbers ("Actually, make it 6 pm.")
    args = cwn(" ".join(str(v) for v in c["arguments"].values()))
    # 10-21: a search may take its words from the earlier turns ("When was she born?" -> "Jane Austen birth date"),
    # other tools only for a message that reads as a follow-up ("Send the same message to Tom.", "Now into French.");
    # a translate of an earlier message ("What is my name?" -> translate "I am Maria") stays groundless
    hist_ok = name == "search" or re.search(_FOLLOW, question, re.I)
    return not (cwn(question) & args or (hist_ok and cwn(context) & args))


_FOLLOW = (r"\b(?:it|that|this|those|them|same|again|instead|actually|also|too|another|other|meant|now|and|what\s+about|"
           r"how\s+about)\b")


def route(question, say=None):
    """A tool call for an explicit request the model answered without a tool, or None."""
    q = question.strip()
    lead = r"^\s*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    m = re.match(lead + r"(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google)\s+(.+?)[.?!]?\s*$", q, re.I)
    if m:
        return {"name": "search", "arguments": {"query": m.group(1)}}
    import tools_exec
    if re.search(r"\bweather\b|\bforecast\b|\b(?:rain|snow)(?:ing)?\b", q, re.I) and _place(q):
        return {"name": "get_weather", "arguments": {"location": _place(q), "unit": _unit(q) or tools_exec.home_unit()}}
    if local_weather(q):                        # 10-21 (Glenn): no place named = this computer's city, home units
        return {"name": "get_weather", "arguments": {"location": tools_exec.HOME["city"],
                                                     "unit": _unit(q) or tools_exec.home_unit()}}
    if re.search(r"\bwhat\s+time\s+is\s+it\b|\bthe\s+time\s+in\b|\bcurrent\s+time\b", q, re.I) and _place(q):
        return {"name": "get_time", "arguments": {"timezone": _place(q)}}
    if local_time(q):                           # 10-21 (Glenn): "What time is it?" = here
        return {"name": "get_time", "arguments": {"timezone": tools_exec.HOME["tz"]}}
    # facts about a person / work / event: grounded in a lookup, not the 1B model's memory (it gave Marie Curie the
    # wrong birth year and Hamlet two premiere dates)
    # 10-18: + "who is / who's" (the model said "Joe Biden" for the current president: the search reads the office
    # article's incumbent) and measured facts ("how far is the Moon": the model put the Moon in orbit around the Sun)
    # 10-21 (chat brain): not for a pronoun with no name ("When was she born?", "How tall is it?"): a search for that
    # sentence finds nothing; the model resolves the pronoun from the conversation
    # (and not about the user: "When was I born?" is in the conversation, not on the web)
    pron = (re.search(r"\b(?:he|she|it|they|him|her|them|his|its|their|that|this|there)\b", q, re.I)
            and not re.search(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z'-]{2,}", q)) or re.search(r"\b(?:i|my|we|our)\b", q, re.I)
    if not pron and re.match(lead + r"(?:tell\s+me\s+about|what\s+do\s+you\s+know\s+about|who\s+(?:wrote|was|were|is|are|invented|"
                r"discovered|painted|composed|founded|directed|built|designed|runs|leads)|who's|when\s+(?:was|were|did)|"
                r"how\s+(?:far|tall|high|deep|long|old|big|large|heavy|fast|many\s+people)\s+(?:is|are|was|were|does|did)|"
                r"what\s+is\s+the\s+(?:population|distance|height|area|diameter|depth|length|boiling\s+point|"
                r"melting\s+point|density)\s+of)\s+\S", q, re.I):
        return {"name": "search", "arguments": {"query": _topic(q)}}
    # 10-21 (Glenn): "Define 'zarnok'." was answered from memory ("a Polish word that means to be happy") -> the
    # dictionary, which says when it has no entry (one-word terms only; curly quotes as typed on a phone / Mac)
    w = re.match(lead + r"(?:define|(?:what\s+is\s+|what's\s+)?the\s+(?:meaning|definition)\s+of|what\s+does)\s+"
                 r"(?:the\s+word\s+)?['\"‘’“”]?([A-Za-z][\w-]*)['\"‘’“”]?(?:\s+mean)?\s*[.?!]?\s*$", q, re.I)
    if w and (not re.match(lead + r"what\s+does", q, re.I) or re.search(r"\bmean\s*[.?!]?\s*$", q, re.I)):
        return {"name": "define", "arguments": {"term": w.group(1)}}
    r = _reminder(q)
    if r:
        return {"name": "set_reminder", "arguments": {"text": r[0], "datetime": r[1] or "unspecified"}}
    e = _arith(q)
    if e:
        return {"name": "calculator", "arguments": {"expression": e}}
    return None


def fix_spelling(reply, context, source=""):
    """Words of the reply the model mis-copied from the question / tool result ("Ephemereral" for "ephemeral"):
    a word of 6+ letters that is not in the context, not an English word (WordNet), and within edit distance of a
    context word is replaced by it. 10-21: `source` = the tool result, whose spelling wins over the question's
    ("Who was Einstien?" -> the reply copied "Einstien" from the question; the article says "Einstein").
    Returns (reply, ["Ephemereral -> ephemeral", ...])."""
    import difflib
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return reply, []
    ctx = {w.lower(): w for w in re.findall(r"[A-Za-z][A-Za-z'-]{5,}", context)}
    src = {w.lower(): w for w in re.findall(r"[A-Za-z][A-Za-z'-]{5,}", source)}
    fixes = []

    def rep(m):
        w = m.group(0)
        if wn.synsets(w.lower()):
            return w
        if w.lower() in ctx and (w.lower() in src or not src):
            return w
        near = difflib.get_close_matches(w.lower(), list(src), n=1, cutoff=0.8) if w.lower() not in src else []
        if not near and w.lower() in ctx:
            return w
        near = near or difflib.get_close_matches(w.lower(), list(ctx), n=1, cutoff=0.8)
        if not near:
            return w
        new = src.get(near[0]) or ctx[near[0]]
        new = new[0].upper() + new[1:] if w[0].isupper() else new
        fixes.append(f"{w} -> {new}")
        return new
    return re.sub(r"[A-Za-z][A-Za-z'-]{5,}", rep, reply), fixes


_SOCIAL = [(r"^(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening)|greetings)\b[\s!.,]*(?:there)?[\s!.]*$",
            "Hello! Ask me a question, or ask me to look something up, check the weather, convert units or set a reminder."),
           (r"^(?:thanks|thank\s+you|thx|cheers)\b[\s\w!.,]{0,20}$", "You're welcome!"),
           (r"^(?:bye|goodbye|see\s+you)\b[\s\w!.,]{0,20}$", "Goodbye!"),
           (r"^(?:hi|hello|hey)?[\s,!]*(?:who|what)\s+are\s+you\??$",
            "I am Neuro, a 1-billion-parameter language model with a spiking memory layer, running on this computer. "
            "I can answer questions and use tools: search, weather, time, calculator, unit conversion, dictionary, "
            "translation, reminders and messages."),
           # 10-20: a market prediction has no answer to look up ("What will the stock market do next week?" searched
           # "the current price of a stock" -> CAPE "is expected to be higher next week")
           (r"^(?=.*\b(?:will|going\s+to|predict\w*|forecast\w*|outlook)\b)(?=.*(?:\bstock\s+market|\bstocks?\b|\bshares?\b|"
            r"\bthe\s+market\b|\bbitcoin\b|\bcrypto\w*|\bs&p\b|\bs\s*&\s*p\s*500\b|\bdow\b|\bnasdaq\b|\bftse\b|"
            r"\b(?:stock|share|oil|gold)\s+prices?\b|\binterest\s+rates?\b)).*$",
            "Nobody can reliably predict what the market will do next week, and I can't give investment advice. "
            "I can look up how it has done in the past or explain terms like CAPE."),
           # 10-20: an action no tool can do ("Order a pizza for me." -> send_message "I got the pizza" to "the team")
           (r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:i(?:\s+(?:want|need|would\s+like)|['’]d\s+like)\s+"
            r"(?:you\s+)?to\s+)?(?:order|buy|purchase|book|reserve|pay|call|phone|ring|hire|rent)\b",
            "I can't order, buy or book things. I can look things up, check the weather, convert units, set reminders "
            "or draft a message.")]


_PY_ASK = (r"^(?=.*\bpython\b)(?=.*\b(?:write|define|create|make|implement|show\s+me|give\s+me|i\s+need)\b)"
           r"(?=.*\b(?:function|code|script|program)\b)")
PY_OFF = "I don't have the Python skill loaded. Turn on the Python brain and ask again."


def needs_python(question):
    """10-20 (Glenn): a request to WRITE Python code ("Using Python, define a function that ...") -- with the
    Python brain off the engine answers PY_OFF instead of the base model's attempt (0% of its code passed the quiz)."""
    return re.match(_PY_ASK, question.strip(), re.I) is not None


_PY_FNS = None
PY_UNTAUGHT = ("That is not part of my training set. My Python brain was taught about 30 small functions, such as "
               "check_palindrome, count_vowels, fizzbuzz and fibonacci.")


def python_untaught(question):
    """10-21 (Glenn): a code request that names a function the Python brain was not taught ("call it is_power_of_two")
    -> True. Taught = scripts/demo/python_functions.json (the names of scripts/gen_pybrain_data.py, controls left out).
    A request that names no function is left to the model."""
    global _PY_FNS
    m = re.search(r"\b(?:call(?:ed)?\s+it|name(?:d)?\s+it|called|named)\s+[`'\"]?([A-Za-z_]\w*)", question)
    if not m:
        return False
    if _PY_FNS is None:
        import json, os
        _PY_FNS = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_functions.json"))))
    return m.group(1).lower() not in _PY_FNS


def off_topic(question, text):
    """10-21 (Glenn): a search result that does not mention a name the question asks about ("population of Zorbania"
    -> the article "Population without double counting"; "the Zelphar Empire" -> "The Fall of the Roman Empire") ->
    the missing name, else None. Every word not in WordNet (4+ letters: "Zorbania", "Glorbistan") must be in the result,
    and at least one capitalised word inside the question ("How far is the Moon from Earth?" -> "Moon: averaging around
    378,000 km" is fine); a near spelling counts ("Einstien" / "Einstein")."""
    import difflib
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    q = re.sub(r"[`'\"“”‘’]", " ", question.strip())
    caps = {w for w in re.findall(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z'-]{3,}", q) if w.lower() not in _STOP}
    rare = {w for w in re.findall(r"\b[A-Za-z][a-z'-]{3,}\b", q) if w.lower() not in _FUNC | _STOP
            and not wn.synsets(w.lower())}
    have = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]+", str(text))}
    seen = lambda w: bool(difflib.get_close_matches(w.lower(), have, 1, 0.8))
    for w in sorted(rare):
        if not seen(w):
            return w
    if caps and not any(seen(w) for w in caps):
        return sorted(caps)[0]
    return None


NOT_UNDERSTOOD ="I don't understand. Could you say that another way?"
_FUNC = set("""a an the i i'm i'd i'll me my mine you you're your yours he him his she her hers it it's its we us our
they them their this that these those what what's who who's whom whose which when where why how is am are was were be
been being do does did done have has had will would shall should can could may might must of to in on at by for with
from into onto about over under up down out off and or but if then than so not no yes please tell give show let let's
there here also just very much many more most some any all each every one two own same such too again ever never
don't can't won't isn't aren't didn't doesn't""".split())


def gibberish(question):
    """10-21 (Glenn): a message that is mostly not English words ("blorf the zimzam quickly?", "asdfgh qwerty") -> True,
    so the demo says NOT_UNDERSTOOD instead of the 1B model making something up. Words = lower-case words plus the
    first word (capitalised words inside the message are names and are not counted), function words left out; True
    when more than half of the rest are not in WordNet ("Define ubiquitus" = 1 of 2: left to the tools)."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return False
    q = question.strip()
    if "\n" in q or re.search(r"[{}=;<>\[\]]", q):          # pasted code / text blocks
        return False
    q = re.sub(r"(['\"“‘])+[^'\"“”‘’]+?['\"”’]+", " ", q)  # quoted text (foreign phrases to translate)
    toks = re.findall(r"[A-Za-z][A-Za-z']*", q)
    words = [w.lower() for i, w in enumerate(toks) if (i == 0 or not w[0].isupper()) and w.lower() not in _FUNC]
    unknown = sum(not (wn.synsets(w.replace("'", "")) or wn.synsets(w.rstrip("s"))) for w in words)
    return unknown * 2 > len(words)


_NUTRIENT = r"\b(?:protein|calories?|kcal|carbs?|carbohydrates?|energy|nutrition|nutrients?)\b"
_COOKED = r"\b(?:cooked|boiled|fried|roasted|baked|grilled|steamed|dried|canned|frozen|smoked)\b"
_FOODS = None
HEALTH_UNTAUGHT = ("That is not part of my training set. My health brain covers 150 plain raw foods from USDA data, "
                   "such as raw lentils, spinach or tilapia.")


HEALTH_OFF = "I don't have the health skill loaded. Turn on the health brain and ask again."


def needs_health(question):
    """A nutrient-amount question ("Give me the kcal for 100 g of raw pumpkin."). 10-21 (Glenn): with the health brain
    off the base model made one up (calculator "100 * 0.5" -> "50 kcal"; search -> Kale, the moon of Jupiter), so the
    engine answers HEALTH_OFF, as PY_OFF for code."""
    q = question.lower()
    return bool(re.search(_NUTRIENT, q) and re.search(r"\b(?:how\s+(?:much|many)|content|grams?|\d+\s*g|contains?|has|"
                                                      r"have|in\s+(?:a|an|one|\d+))\b", q))


def untaught_food(question):
    """10-20 (Glenn): a nutrient question about a food the health brain was not taught ("100 grams of beef" -> it
    invented 3 g, 2 g, 4.2 g) or a cooked / dried one (its values are raw) -> True. Taught = the 150 foods of
    health_foods.json (temp/n70_mk_health_foods.py); "banana" / "grape" match "bananas" / "grapes"."""
    global _FOODS
    q = question.lower()
    if not needs_health(question):
        return False                            # not an amount question ("What is protein?")
    if re.search(_COOKED, q):
        return True
    if _FOODS is None:
        import json, os
        names = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "health_foods.json")))
        _FOODS = [re.compile(r"\b" + re.escape(re.sub(r"(?:es|s)$", "", f)) + r"(?:e?s|e)?\b") for f in names]
    return not any(p.search(q) for p in _FOODS)


def social(question):
    """A fixed reply for pure small talk (greeting / thanks / goodbye / who are you), else None. The 1B model's
    small talk is weak (it called a translator for "hi"); the page labels this reply as the backend's."""
    q = question.strip().lower()
    for pat, reply in _SOCIAL:
        if re.match(pat, q, re.I):
            return reply
    return None
