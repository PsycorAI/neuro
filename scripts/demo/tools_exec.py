"""demo/tools_exec.py -- 2026-10-12 (Neuro 57, docs/demo_v4_plan_2026_10_12.md D2). Executors for the 9 tools of
training_data/posttrain/tools.json. One entry point:

    run(name, arguments, offline=False) -> {"ok": bool, "text": str, "source": "local|live|offline|simulated"}

`text` is the short plain-text result the model reads in its second pass (### Input: Result from <tool>: <text>).
Every network call has a 4 s timeout and falls back to the offline path; nothing here can raise to the caller.
  local      calculator (ast evaluator, never eval), unit_convert (own table), get_time (zoneinfo)
  live       search (Wikipedia REST), get_weather (Open-Meteo, no key); both cached to logs/demo_cache.json
  offline    search (SQLite FTS5 pack data/demo/search_pack.db), define (WordNet via nltk), translate (opus-mt)
  simulated  set_reminder, send_message (stored in logs/demo_outbox.json; nothing leaves the machine)"""
import ast, difflib, json, math, operator, os, re, sqlite3, time, datetime, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "logs", "demo_cache.json")
OUTBOX = os.path.join(ROOT, "logs", "demo_outbox.json")
PACK = os.path.join(ROOT, "data", "demo", "search_pack.db")
TIMEOUT = 4


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def _get(url):
    # 10-18: contact in the User-Agent (Wikimedia's API policy; identified clients get the normal rate limits)
    req = urllib.request.Request(url, headers={"User-Agent": "neuro-demo/0.2 (https://psycor.ai/; local research demo)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


# --------------------------------------------------------------- home location (10-21, Glenn)
_IMPERIAL = {"US", "LR", "MM"}                 # countries on miles / Fahrenheit


def _home():
    """This computer's location: time zone from the system, its country from the tz database's zone.tab (offline),
    city from scripts/demo/home.json ({"city": "Seattle, Washington"}; empty = the time zone's city). The online demo
    will need the visitor's location instead (to-do 10-21)."""
    tz = os.environ.get("DEMO_TZ", "")
    if not tz:
        try:
            tz = open("/etc/timezone").read().strip()
        except OSError:
            p = os.path.realpath("/etc/localtime")
            tz = p.split("zoneinfo/", 1)[1] if "zoneinfo/" in p else "UTC"
    country = ""
    try:
        for line in open("/usr/share/zoneinfo/zone.tab", encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if not line.startswith("#") and len(f) > 2 and f[2] == tz:
                country = f[0]
                break
    except OSError:
        pass
    city = str(_load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "home.json"), {}).get("city", "")).strip()
    return dict(tz=tz, country=country, city=city or tz.rsplit("/", 1)[-1].replace("_", " "), city_set=bool(city),
                imperial=country in _IMPERIAL)


HOME = _home()


def home_unit():
    return "fahrenheit" if HOME["imperial"] else "celsius"


def _num(x):
    """Short human number: integers without a decimal point, else up to 6 significant digits."""
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        x = int(x)
    return f"{x:,}" if isinstance(x, int) else f"{x:.6g}"


# ------------------------------------------------------------------ calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_FUN = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10, "sin": math.sin,
        "cos": math.cos, "tan": math.tan, "exp": math.exp, "min": min, "max": max, "floor": math.floor,
        "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e}


def _ev(n):
    if isinstance(n, ast.Expression):
        return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(b) > 200 or abs(a) > 1e6):
            raise ValueError("power too large")
        return _BIN[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        return -_ev(n.operand) if isinstance(n.op, ast.USub) else _ev(n.operand)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords:
        return _FUN[n.func.id](*[_ev(a) for a in n.args])
    if isinstance(n, ast.Name) and n.id in _CONST:
        return _CONST[n.id]
    raise ValueError("not arithmetic")


def calculator(expression):
    e = str(expression).strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", e, flags=re.I)      # "17% of 240"
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", e)           # "50%" but not the modulo "7 % 3"
    try:
        v = _ev(ast.parse(e, mode="eval"))
        if isinstance(v, float):
            v = round(v, 10)
        return dict(ok=True, text=_num(v), source="local")
    except ZeroDivisionError:
        return dict(ok=False, text="division by zero", source="local")
    except Exception:
        return dict(ok=False, text=f"could not evaluate: {expression}", source="local")


# ---------------------------------------------------------------- unit_convert
_U = {  # unit -> (dimension, factor to the SI base)
    "length": {"m": 1, "meter": 1, "metre": 1, "km": 1e3, "kilometer": 1e3, "kilometre": 1e3, "cm": 1e-2,
               "centimeter": 1e-2, "mm": 1e-3, "millimeter": 1e-3, "mile": 1609.344, "mi": 1609.344, "yard": 0.9144,
               "yd": 0.9144, "foot": 0.3048, "feet": 0.3048, "ft": 0.3048, "inch": 0.0254, "inches": 0.0254,
               "in": 0.0254, "nautical mile": 1852},
    "mass": {"kg": 1, "kilogram": 1, "g": 1e-3, "gram": 1e-3, "mg": 1e-6, "milligram": 1e-6, "lb": 0.45359237,
             "lbs": 0.45359237, "pound": 0.45359237, "oz": 0.028349523125, "ounce": 0.028349523125,
             "ton": 907.18474, "tonne": 1000, "metric ton": 1000, "stone": 6.35029318},
    "volume": {"l": 1, "liter": 1, "litre": 1, "ml": 1e-3, "milliliter": 1e-3, "millilitre": 1e-3,
               "gallon": 3.785411784, "gal": 3.785411784, "quart": 0.946352946, "pint": 0.473176473,
               "cup": 0.2365882365, "fluid ounce": 0.0295735296, "fl oz": 0.0295735296, "tablespoon": 0.0147867648,
               "tbsp": 0.0147867648, "teaspoon": 0.00492892159, "tsp": 0.00492892159, "cubic meter": 1000},
    "speed": {"m/s": 1, "meters per second": 1, "km/h": 1 / 3.6, "kph": 1 / 3.6, "kilometers per hour": 1 / 3.6,
              "mph": 0.44704, "miles per hour": 0.44704, "knot": 0.514444, "ft/s": 0.3048},
    "area": {"square meter": 1, "m2": 1, "sq m": 1, "square kilometer": 1e6, "km2": 1e6, "square foot": 0.09290304,
             "square feet": 0.09290304, "sq ft": 0.09290304, "acre": 4046.8564224, "hectare": 1e4,
             "square mile": 2589988.110336, "square inch": 0.00064516, "square yard": 0.83612736},
    "time": {"second": 1, "s": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "h": 3600,
             "day": 86400, "week": 604800, "year": 31557600, "millisecond": 1e-3, "ms": 1e-3},
    "data": {"byte": 1, "b": 1, "kilobyte": 1e3, "kb": 1e3, "megabyte": 1e6, "mb": 1e6, "gigabyte": 1e9, "gb": 1e9,
             "terabyte": 1e12, "tb": 1e12, "bit": 0.125},
    "energy": {"joule": 1, "j": 1, "kilojoule": 1e3, "kj": 1e3, "calorie": 4.184, "cal": 4.184, "kilocalorie": 4184,
               "kcal": 4184, "kwh": 3.6e6, "kilowatt hour": 3.6e6, "watt hour": 3600, "wh": 3600},
}
_TEMP = {"celsius": "c", "c": "c", "°c": "c", "centigrade": "c", "fahrenheit": "f", "f": "f", "°f": "f",
         "kelvin": "k", "k": "k"}


def _unit(u):
    u = str(u).strip().lower().replace("-", " ")
    u = re.sub(r"^degrees? ", "", u)
    for cand in (u, u.rstrip("s"), u[:-2] if u.endswith("es") else u):
        if cand in _TEMP:
            return "temperature", cand
        for dim, tab in _U.items():
            if cand in tab:
                return dim, cand
    return None, u


def unit_convert(value, from_unit, to_unit):
    try:
        v = float(value)
    except Exception:
        return dict(ok=False, text=f"not a number: {value}", source="local")
    (d1, a), (d2, b) = _unit(from_unit), _unit(to_unit)
    if d1 is None or d2 is None or d1 != d2:
        return dict(ok=False, text=f"cannot convert {from_unit} to {to_unit}", source="local")
    if d1 == "temperature":
        k = {"c": v + 273.15, "f": (v - 32) * 5 / 9 + 273.15, "k": v}[_TEMP[a]]
        out = {"c": k - 273.15, "f": (k - 273.15) * 9 / 5 + 32, "k": k}[_TEMP[b]]
    else:
        out = v * _U[d1][a] / _U[d1][b]
    return dict(ok=True, text=f"{_num(v)} {from_unit} = {_num(round(out, 6))} {to_unit}", source="local")


# -------------------------------------------------------------------- get_time
_TZ = {"est": "America/New_York", "edt": "America/New_York", "eastern": "America/New_York", "new york": "America/New_York",
       "cst": "America/Chicago", "central": "America/Chicago", "chicago": "America/Chicago", "mst": "America/Denver",
       "mountain": "America/Denver", "denver": "America/Denver", "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
       "pacific": "America/Los_Angeles", "los angeles": "America/Los_Angeles", "san francisco": "America/Los_Angeles",
       "seattle": "America/Los_Angeles", "utc": "UTC", "gmt": "UTC", "london": "Europe/London", "paris": "Europe/Paris",
       "berlin": "Europe/Berlin", "madrid": "Europe/Madrid", "rome": "Europe/Rome", "moscow": "Europe/Moscow",
       "tokyo": "Asia/Tokyo", "jst": "Asia/Tokyo", "seoul": "Asia/Seoul", "beijing": "Asia/Shanghai",
       "shanghai": "Asia/Shanghai", "hong kong": "Asia/Hong_Kong", "singapore": "Asia/Singapore",
       "india": "Asia/Kolkata", "ist": "Asia/Kolkata", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
       "dubai": "Asia/Dubai", "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
       "auckland": "Pacific/Auckland", "honolulu": "Pacific/Honolulu", "hawaii": "Pacific/Honolulu",
       "sao paulo": "America/Sao_Paulo", "mexico city": "America/Mexico_City", "toronto": "America/Toronto",
       "cairo": "Africa/Cairo", "johannesburg": "Africa/Johannesburg", "lagos": "Africa/Lagos",
       "bangkok": "Asia/Bangkok", "jakarta": "Asia/Jakarta", "istanbul": "Europe/Istanbul"}
# 10-20: US states (a state with two zones -> the zone most of its people live in) and countries ("What is the time in
# California?" -> the model called America/New_York and "California" was unknown, so the guard could not repair it)
for _zone, _names in {
        "America/New_York": "connecticut delaware florida georgia indiana maine maryland massachusetts michigan "
                            "new hampshire new jersey north carolina ohio pennsylvania rhode island south carolina "
                            "vermont virginia west virginia kentucky washington dc district of columbia",
        "America/Chicago": "alabama arkansas illinois iowa kansas louisiana minnesota mississippi missouri nebraska "
                           "north dakota oklahoma south dakota tennessee texas wisconsin",
        "America/Denver": "colorado idaho montana new mexico utah wyoming", "America/Phoenix": "arizona",
        "America/Los_Angeles": "california nevada oregon washington state", "America/Anchorage": "alaska"}.items():
    for _n in re.findall(r"new \w+|north \w+|south \w+|west \w+|rhode island|district of columbia|washington (?:dc|state)|\w+",
                         _names):
        _TZ[_n] = _zone
_TZ.update({"united kingdom": "Europe/London", "uk": "Europe/London", "england": "Europe/London",
            "britain": "Europe/London", "scotland": "Europe/London", "ireland": "Europe/Dublin", "france": "Europe/Paris",
            "germany": "Europe/Berlin", "spain": "Europe/Madrid", "italy": "Europe/Rome", "netherlands": "Europe/Amsterdam",
            "russia": "Europe/Moscow", "turkey": "Europe/Istanbul", "greece": "Europe/Athens", "egypt": "Africa/Cairo",
            "south africa": "Africa/Johannesburg", "nigeria": "Africa/Lagos", "kenya": "Africa/Nairobi",
            "japan": "Asia/Tokyo", "china": "Asia/Shanghai", "south korea": "Asia/Seoul", "korea": "Asia/Seoul",
            "thailand": "Asia/Bangkok", "indonesia": "Asia/Jakarta", "philippines": "Asia/Manila",
            "vietnam": "Asia/Ho_Chi_Minh", "pakistan": "Asia/Karachi", "israel": "Asia/Jerusalem",
            "uae": "Asia/Dubai", "united arab emirates": "Asia/Dubai", "saudi arabia": "Asia/Riyadh",
            "australia": "Australia/Sydney", "new zealand": "Pacific/Auckland", "canada": "America/Toronto",
            "mexico": "America/Mexico_City", "brazil": "America/Sao_Paulo", "argentina": "America/Argentina/Buenos_Aires",
            "chile": "America/Santiago", "colombia": "America/Bogota", "peru": "America/Lima"})


_ZONES = None                                  # 10-21: available_timezones() reads the tz database (27 ms a call)


def get_time(timezone, offline=False):
    global _ZONES
    from zoneinfo import ZoneInfo, available_timezones
    if _ZONES is None:
        _ZONES = available_timezones()
    q = str(timezone).strip()
    name, where = None, ""
    k = q.lower().replace("_", " ").split(",")[0].strip()
    if q in _ZONES and k not in _TZ:   # 10-20: "EST" is a fixed UTC-5 zone; the table's New York has DST
        name = q
    else:
        name = _TZ.get(k)
        if name is None:                      # "Sao_Paulo" -> America/Sao_Paulo
            hit = [z for z in _ZONES if z.lower().replace("_", " ").endswith("/" + k)]
            name = sorted(hit)[0] if hit else None
        if name is None and not offline:      # 10-20: any other city -> its zone from the weather geocoder (online;
                                              # 10-21: not with the offline switch on)
            try:
                if k not in _GEO:
                    _GEO[k] = _get("https://geocoding-api.open-meteo.com/v1/search?count=1&name="
                                   + urllib.parse.quote(q.split(",")[0].strip()))["results"][0]
                name = _GEO[k].get("timezone")
                where = ", ".join(x for x in (_GEO[k].get("name"), _GEO[k].get("country")) if x)   # which place it found
            except Exception:
                name = None
    if name is None:
        return dict(ok=False, text=f"unknown timezone: {timezone}", source="local")
    now = datetime.datetime.now(ZoneInfo(name))
    return dict(ok=True, text=f"{now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B')} {now.day} "
                              f"({name}{', ' + where if where else ''})",
                source="local")


# ---------------------------------------------------------------------- define
def define(term):
    t = str(term).strip()
    try:
        from nltk.corpus import wordnet as wn
        syn = wn.synsets(t.replace(" ", "_"))
    except Exception:
        return dict(ok=False, text="the offline dictionary is not installed", source="offline")
    if not syn:
        return dict(ok=False, text=f"no definition found for {t}", source="offline")
    # 10-18: the most used sense of the word itself (WordNet lists nouns first: "ephemeral" gave the rare noun sense
    # "anything short-lived, as an insect ..."); ties keep WordNet's order
    w = t.lower().replace(" ", "_")
    syn = sorted(syn, key=lambda s: -sum(l.count() for l in s.lemmas() if l.name().lower() == w))
    pos = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
    return dict(ok=True, text=f"{t} ({pos.get(syn[0].pos(), syn[0].pos())}): {syn[0].definition()}", source="offline")


# ---------------------------------------------------------------------- search
def _clip(s, n=420):
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    return cut[:cut.rfind(". ") + 1] if ". " in cut[80:] else cut.rsplit(" ", 1)[0] + "..."


def search_offline(query):
    if not os.path.exists(PACK):
        return dict(ok=False, text="no result found (offline pack not installed)", source="offline")
    words = re.findall(r"[A-Za-z0-9]{3,}", str(query))
    stop = {"the", "what", "who", "when", "where", "which", "was", "were", "are", "is", "did", "does", "how", "and",
            "for", "that", "this", "with", "from", "name", "has", "have", "had"}
    words = [w for w in words if w.lower() not in stop][:8]
    if not words:
        return dict(ok=False, text="no result found", source="offline")
    con = sqlite3.connect(PACK)
    try:
        for q in (" ".join(words), " OR ".join(words)):
            row = con.execute("select title, body from docs where docs match ? order by bm25(docs, 6.0, 1.0) limit 1",
                              (q,)).fetchone()
            if row:
                return dict(ok=True, text=_clip(f"{row[0]}: {row[1]}"), source="offline")
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return dict(ok=False, text="no result found", source="offline")


_RECENT = {}                                   # key -> (time, result): in-process recent results (search, weather)
_GEO = {}                                      # place -> Open-Meteo geocoding hit
_MEASURE = {"far": r"distance|kilomet|\bkm\b|miles|\bmi\b", "distance": r"distance|kilomet|\bkm\b|miles",
            "tall": r"tall|height|high|metres|meters|feet|\bft\b", "high": r"tall|height|high|metres|meters|feet",
            "height": r"tall|height|high|metres|meters|feet", "deep": r"deep|depth", "depth": r"deep|depth",
            "long": r"long|length|kilomet|\bkm\b|miles", "length": r"long|length|kilomet|\bkm\b|miles",
            "old": r"\bage\b|years|born|founded|formed|built|established", "big": r"area|square|size|km2|largest",
            "large": r"area|square|size|km2|largest", "area": r"area|square|km2", "diameter": r"diameter|across|wide",
            "heavy": r"mass|weigh|\bkg\b|tonnes|tons", "fast": r"speed|km/h|mph|per hour|per second",
            "many": r"population|million|billion|number", "population": r"population|inhabitants|million|billion",
            "boiling point": r"\bboil", "melting point": r"\bmelt", "density": r"densit|g/cm"}
_WAPI = "https://en.wikipedia.org/w/api.php?format=json&formatversion=2&redirects=1&action=query&"
_OFFICE = (r"(?:who\s+is\s+|who's\s+)?(?:the\s+)?(?:current\s+|present\s+)?((?:president|vice\s+president|prime\s+minister|"
           r"chancellor|king|queen|monarch|pope|governor|mayor|premier|secretary[\s-]general|first\s+minister|"
           r"chief\s+executive|ceo|speaker)\s+of\s+.+?)\s*(?:right\s+now|now|today|currently)?[?.!]?$")


def _measure_score(s, kw):
    """2 = a number (not a percentage) within 60 characters of the measure's word, 1 = both somewhere, 0 = not."""
    nums = [m.start() for m in re.finditer(r"\d[\d,.]*(?![\d,.]*\s*%)", s)]
    keys = [m.start() for m in re.finditer(kw, s, re.I)]
    if not nums or not keys:
        return 0
    return 2 if any(abs(n - k) < 60 for n in nums for k in keys) else 1


_WD_PROP = {"boiling point": "P2102", "melting point": "P2101", "density": "P2054"}


def _wikidata(title, what):
    """10-20: a boiling / melting point or density from Wikidata (iron's is only in its infobox; the article text
    gives pig iron's): the preferred value, else the only one, in °C (+ °F) or g/cm3; None when the values disagree
    (water's density at 26 temperatures) or there are none."""
    try:
        pg = _get(_WAPI + "prop=pageprops&ppprop=wikibase_item&titles=" + urllib.parse.quote(title))["query"]["pages"][0]
        qid = pg.get("pageprops", {}).get("wikibase_item")
        if not qid:
            return None
        cl = _get("https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity=" + qid + "&property="
                  + _WD_PROP[what])["claims"].get(_WD_PROP[what], [])
        vals = []
        for c in cl:
            v = c["mainsnak"].get("datavalue", {}).get("value") or {}
            a, u = float(v.get("amount", "nan")), str(v.get("unit", "")).rsplit("/", 1)[-1]
            a = {"Q25267": a, "Q11579": a - 273.15}.get(u) if what != "density" else {"Q13147228": a, "Q844211": a / 1000}.get(u)
            if a is not None:              # °C / kelvin, g/cm3 / kg/m3 (a °F duplicate of the °C value is skipped)
                vals.append((c.get("rank") == "preferred", round(a, 3)))
        vals = [a for p, a in vals if p] or [a for p, a in vals]
        if len(set(vals)) != 1:
            return None
        a = vals[0]
        if what == "density":
            return f"{_num(a)} g/cm3"
        c, f = f"{_num(round(a, 1))} °C", f"{_num(round(a * 9 / 5 + 32))} °F"
        return f"{f} ({c})" if HOME["imperial"] else f"{c} ({f})"      # 10-21: the home unit first
    except Exception:
        return None


_NUT_WORD = {"protein": "protein", "carb": r"carbohydrates?", "fat": "fat"}


def _nutrition(q):
    """10-20: "How much protein does 100 grams of raw pumpkin have" -> the article's standard nutrition line ("Pumpkin
    is 92% water, 6.5% carbohydrate, 0.1% fat and 1% protein"; "In a 100-gram amount ... 26 kilocalories"), which is
    past the intro (the model copied "100 grams" from the question as the answer). x% = x grams per 100 grams. None =
    not a nutrient-amount question; a failed result when the article gives no such line (Beef)."""
    m = re.search(r"\b(protein|calories?|kcal|carbs?|carbohydrates?|fat)\b", q, re.I)
    if not m or not re.search(r"how\s+(?:much|many)|content|\bin\s+(?:a|an|\d)|\d+\s*(?:g|grams?)\b", q, re.I):
        return None
    w = m.group(1).lower()
    kind = "kcal" if w.startswith(("calor", "kcal")) else "carb" if w.startswith("carb") else w
    food = re.sub(r"\b(?:how|much|many|protein|content|calories?|kcal|carbs?|carbohydrates?|fat|does|do|is|are|in|"
                  r"have|has|contains?|there|grams?|g|of|the|a|an|per|amount|what|s|\d+)\b|[^\w\s]", " ", q, flags=re.I)
    food = re.sub(r"\s+", " ", food).strip()
    if not food:
        return None
    hits = _get(_WAPI + "list=search&srlimit=1&srsearch=" + urllib.parse.quote(food))["query"]["search"]
    if not hits:
        return None
    title = hits[0]["title"]
    j = _get(_WAPI + "prop=extracts&explaintext=1&titles=" + urllib.parse.quote(title))
    sents = [re.sub(r"=+[^=]+=+\s*", "", s).strip()
             for s in re.split(r"(?<=[.!?])\s+", j["query"]["pages"][0].get("extract") or "")]
    if kind == "kcal":
        rx = r"(\d[\d,.]*)\s*(?:kilocalories|calories|kcal)\b"
        s = next((s for s in sents if re.search(rx, s) and "100" in s), None)
        if s:
            n = re.search(rx, s).group(1)
            return dict(ok=True, source="live",
                        text=_clip(f"{title}: about {n} calories (kcal) per 100 grams (Wikipedia: \"{s}\")", 520))
        what = "calories"
    else:
        rx = r"(\d[\d.]*)\s*%\s*" + _NUT_WORD[kind]
        s = next((s for s in sents if re.search(rx, s)), None)
        word = {"carb": "carbohydrate"}.get(kind, kind)
        if s:
            n = re.search(rx, s).group(1)
            return dict(ok=True, source="live", text=_clip(f"{title}: {n}% {word} = about {n} gram{'' if n == '1' else 's'} of {word} per 100 "
                                                           f"grams (Wikipedia: \"{s}\")", 520))
        what = f"{word} content"
    return dict(ok=False, text=f"the Wikipedia article {title} does not give the {what} per 100 grams", source="live")


_ASK_STOP = {"the", "what", "when", "which", "does", "did", "was", "were", "are", "is", "how", "many", "much", "long",
             "time", "year", "date", "day", "tomorrow", "today", "now", "usually", "and", "for", "with", "from", "that",
             "this", "there", "have", "has", "will", "can", "you", "tell"}


def _answer_sentence(question, title, intro):
    """10-20: "What time does the NYSE open tomorrow?" -> the intro has no time and the model invented 11:00 AM; the
    hours are deep in the article ("The NYSE is open for trading ... from 9:30 am - 4:00 pm ET"). For when / what time
    / what year / how many / how long questions: the sentence with the right kind of number and most of the
    question's words, from the 12-sentence lead, else the whole article (no See also / References lists). Only when the
    intro has no number of that kind at all ("When did World War I end" picked a See-also line, "When was the Mona
    Lisa painted" a 2007 sentence, while the intro had the years). None = keep the intro; False = no sentence anywhere
    has that kind of number with a question word."""
    m = re.match(r"\s*(?:(what\s+time|at\s+what\s+time)|(when|what\s+year|which\s+year|in\s+what\s+year|what\s+date)|"
                 r"(how\s+(?:many|much|long)))\b", question, re.I)
    if not m:
        return None
    num = (r"\d{1,2}(?::\d\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b" if m.group(1) else
           r"\b(?:1\d{3}|20\d\d)\b" if m.group(2) else r"\d")
    kws = {w[:5] for w in re.findall(r"[a-z0-9]{3,}", question.lower()) if w not in _ASK_STOP}
    tw = {w[:5] for w in re.findall(r"[a-z0-9]{3,}", title.lower())}
    # a question word that is not in the title ("painted") counts 3x a title word ("Mona Lisa" is in every sentence:
    # "acquired the Mona Lisa after Leonardo's death in 1519" beat "painted between 1503 and 1506")
    score = lambda s: (sum(1 if k in tw else 3 for k in kws & {w[:5] for w in re.findall(r"[a-z0-9]{3,}", s.lower())})
                       if re.search(num, s, re.I) else 0)
    split = lambda t: [x.strip() for x in re.split(r"(?<=[.!?])\s+", re.sub(r"=+[^=\n]+=+", " ", t)) if x.strip()]
    if any(re.search(num, s, re.I) for s in split(intro)):
        return None                             # the intro has such a number: keep it
    for extra in ("exintro=1&exsentences=12&", ""):
        j = _get(_WAPI + "prop=extracts&" + extra + "explaintext=1&titles=" + urllib.parse.quote(title))
        txt = re.split(r"\n=+\s*(?:See also|References|Notes|Further reading|External links|Bibliography)\s*=+",
                       j["query"]["pages"][0].get("extract") or "")[0]
        best = max(split(txt), key=score, default="")
        if score(best):
            return best
    return False


def _incumbent(office):
    """10-18: "president of the United States" -> "Donald Trump", read from the `incumbent` field of the office
    article's infobox (the summary text never names the holder; the 1B model answered "Joe Biden" from memory)."""
    t = re.sub(r"\s+", " ", office).strip()
    t = t[0].upper() + t[1:]
    ceo = re.match(r"(?:ceo|chief executive(?: officer)?)\s+of\s+(?:the\s+)?(.+)", t, re.I)
    if ceo:                                   # companies name the CEO in key_people: "[[Tim Cook]] ([[CEO]])"
        hit = _get(_WAPI + "list=search&srlimit=1&srsearch=" + urllib.parse.quote(ceo.group(1) + " company"))["query"]["search"]
        if not hit:
            return None
        pg = _get(_WAPI + "prop=revisions&rvprop=content&rvslots=main&rvsection=0&titles=" + urllib.parse.quote(hit[0]["title"]))
        pg = pg["query"]["pages"][0]
        txt = pg["revisions"][0]["slots"]["main"]["content"] if "revisions" in pg else ""
        kp = re.search(r"\|\s*key_people\s*=(.+?)\n\s*\|\s*\w+\s*=", txt, re.S)
        m = kp and re.search(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\][^\[\n]{0,30}?\([^)\n]{0,60}?(?:CEO|chief executive)",
                             kp.group(1), re.I)
        return (m.group(1).strip(), f"CEO of {pg['title']}") if m else None
    for title in (t, None):
        if title is None:                     # the office article by search ("prime minister of britain")
            hit = _get(_WAPI + "list=search&srlimit=1&srsearch=" + urllib.parse.quote(t))["query"]["search"]
            if not hit:
                return None
            title = hit[0]["title"]
        pg = _get(_WAPI + "prop=revisions&rvprop=content&rvslots=main&rvsection=0&titles=" + urllib.parse.quote(title))
        pg = pg["query"]["pages"][0]
        if "revisions" not in pg:
            continue
        m = re.search(r"\|\s*incumbent\s*=\s*\[\[([^\]|]+)", pg["revisions"][0]["slots"]["main"]["content"])
        if m:
            return m.group(1).strip(), pg["title"]
    return None


def search(query, offline=False, question=None):
    q = str(query).strip()
    # 10-18: "the latest news about the Mars rover" -> "the Mars rover" (Wikipedia has no news; the topic finds the article)
    q = re.sub(r"^(?:the\s+)?(?:(?:latest|recent|current|newest)\s+)?(?:news|updates?|information|info|facts)\s+(?:about|on|regarding|for)\s+",
               "", q, flags=re.I) or str(query).strip()
    cache = _load(CACHE, {})
    key = "search|" + q.lower() + ("|" + str(question).strip().lower() if question else "")   # 10-20: answer depends on it
    busy = False
    if not offline:
        hit = _RECENT.get(key)                 # 10-18: same question within 15 min -> no network (faster; Wikimedia
        if hit and time.time() - hit[0] < 900:  # rate-limits bursts of anonymous API calls with HTTP 429)
            return dict(hit[1])
        try:
            m = re.match(_OFFICE, q, re.I)
            inc = _incumbent(m.group(1)) if m else None
            if inc:
                res = dict(ok=True, text=f"{inc[0]} is the current {inc[1]} (incumbent, Wikipedia).", source="live")
                cache[key] = res["text"]; _save(CACHE, cache); _RECENT[key] = (time.time(), res)
                return res
            nut = _nutrition(q)
            if nut:
                if nut["ok"]:
                    cache[key] = nut["text"]; _save(CACHE, cache); _RECENT[key] = (time.time(), nut)
                return nut
            # 10-18: ONE request (search + intro extracts of the top 5; was search, then summary = 2 round trips).
            # A whole question ("Who wrote the novel Pride and Prejudice") ranks "Pride and Prejudice and Zombies"
            # first -> prefer the longest hit whose title is named in the query (plain article first), else the top
            # hit; disambiguation pages ("X may refer to:") are skipped.
            # 10-20: "what is the <measure> of Y" -> search Y ("boiling point of water" found the film "Boiling Point",
            # "density of gold" the article "Density"); the measure is picked from Y's intro below
            # (and "how old is Y": "How old is the Eiffel Tower" ranked the film "Eiffel" first, the tower not in the top 5)
            sq = re.match(r"(?:what\s+is\s+the\s+(?:population|distance|height|area|diameter|depth|length|boiling\s+point|"
                          r"melting\s+point|density)\s+of|how\s+(?:far|tall|high|deep|long|old|big|large|heavy|fast)\s+"
                          r"(?:is|are|was|were))\s+(.+)", q, re.I)
            sq = sq.group(1).rstrip(" ?.!") if sq else q
            j = _get(_WAPI + "generator=search&gsrlimit=5&prop=extracts&exintro=1&explaintext=1&exsentences=4&exlimit=5"
                     "&gsrsearch=" + urllib.parse.quote(sq))
            pages = [p for p in sorted(j["query"]["pages"], key=lambda p: p["index"])
                     if p.get("extract") and not re.search(r"(?:may|can|most commonly|also)\s+refers?\s+to:?\s*$", p["extract"])]
            base = lambda p: re.sub(r"\s*\(.*\)$", "", p["title"])
            # 10-20: a one-word title counts only when the query capitalises it ("capital of France" named the
            # magazine "Capital (French magazine)" over the top hit "List of capitals of France")
            named = [(len(base(p)), p["title"] == base(p), -i, i) for i, p in enumerate(pages)
                     if base(p).lower() in sq.lower()
                     and (" " in base(p) or re.search(r"\b" + re.escape(base(p)) + r"\b", sq))]
            if not named:  # 10-20: a misspelled name ("Treaty of Versaille") -> the title it nearly spells
                qw = sq.lower().split()
                near = lambda t: any(difflib.SequenceMatcher(None, t, " ".join(qw[k:k + len(t.split())])).ratio() >= 0.9
                                     for k in range(len(qw)))
                named = [(len(base(p)), p["title"] == base(p), -i, i) for i, p in enumerate(pages)
                         if " " in base(p) and near(base(p).lower())]
            if pages:
                pg = pages[max(named)[3]] if named else pages[0]
                ext = pg["extract"]
                mq = re.match(r"(?:how\s+(far|tall|high|deep|long|old|big|large|heavy|fast|many)|what\s+is\s+the\s+"
                              r"(population|distance|height|area|diameter|depth|length|boiling\s+point|melting\s+point|"
                              r"density))\b", q, re.I)   # 10-20: + boiling / melting point, density (tungsten)
                if mq:
                    # 10-18: a measured fact -> the sentences that give THAT measure with a number first ("How far is
                    # the Moon" got "It varies over the year"; "75% of its terrain" became "75,000,000 people" when
                    # any number was promoted). Needs the longer intro (the population is sentence 5+).
                    kw = _MEASURE[re.sub(r"\s+", " ", (mq.group(1) or mq.group(2)).lower())]
                    j2 = _get(_WAPI + "prop=extracts&exintro=1&explaintext=1&exsentences=12&titles="
                              + urllib.parse.quote(pg["title"]))
                    ext = j2["query"]["pages"][0].get("extract") or ext
                    sents = re.split(r"(?<=[.!?])\s+", ext)
                    best = max(sents, key=lambda x: _measure_score(x, kw))
                    what = re.sub(r"\s+", " ", (mq.group(1) or mq.group(2)).lower())
                    wd = _wikidata(pg["title"], what) if what in _WD_PROP and _measure_score(best, kw) < 2 else None
                    if wd:
                        res = dict(ok=True, text=f"{pg['title']}: {what} {wd} (Wikidata).", source="live")
                        cache[key] = res["text"]; _save(CACHE, cache); _RECENT[key] = (time.time(), res)
                        return res
                    if _measure_score(best, kw) < 2:
                        # 10-20: not in the intro (water's boiling point, gold's density) -> the whole article; still
                        # not there and the intro never names the measure -> say so (the model invented a number)
                        j3 = _get(_WAPI + "prop=extracts&explaintext=1&titles=" + urllib.parse.quote(pg["title"]))
                        full = re.split(r"(?<=[.!?])\s+", j3["query"]["pages"][0].get("extract") or "")
                        fbest = max(full, key=lambda x: _measure_score(x, kw), default="")
                        if _measure_score(fbest, kw) == 2:
                            best = fbest
                        elif _measure_score(best, kw) == 0:
                            return dict(ok=False, text=f"the Wikipedia article {pg['title']} does not give the {what}",
                                        source="live")
                    if _measure_score(best, kw) == 2:  # only that sentence, cut to its clauses around the number: the
                        cl = re.split(r",\s+(?=[a-z])|;\s+", best)   # 1B model copies the FIRST clause it reads
                        ext = ", ".join(c for c in cl if _measure_score(c, kw)) or best
                    else:
                        ext = " ".join(sorted(sents, key=lambda x: -_measure_score(x, kw)))
                else:
                    ans = _answer_sentence(question or q, pg["title"], ext)
                    if ans is False:
                        return dict(ok=False, text=f"the Wikipedia article {pg['title']} does not say", source="live")
                    ext = ans or ext
                res = dict(ok=True, text=_clip(f"{pg['title']}: {ext}"), source="live")
                cache[key] = res["text"]; _save(CACHE, cache); _RECENT[key] = (time.time(), res)
                return res
        except urllib.error.HTTPError as e:
            busy = e.code == 429 or e.code >= 500
        except Exception:
            pass
    if key in cache:
        return dict(ok=True, text=cache[key], source="offline")
    if busy:                                   # online but throttled: say so (the small offline pack answers a
        return dict(ok=False, text="the search service is busy, please ask again in a few seconds", source="live")
    return search_offline(q)                   # different question: "Moon" -> Phobos)


# ----------------------------------------------------------------- get_weather
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle",
        53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
        73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def get_weather(location, unit=None, offline=False):
    loc = str(location).strip()
    unit = home_unit() if not unit else "fahrenheit" if str(unit).lower().startswith("f") else "celsius"
    cache = _load(CACHE, {})
    key = f"weather|{loc.lower()}|{unit}"
    if not offline:
        hit = _RECENT.get(key)                 # 10-18: a reading from the last 10 min (no network)
        if hit and time.time() - hit[0] < 600:
            return dict(hit[1])
        try:
            gk = loc.split(",")[0].strip().lower()
            if gk not in _GEO:                 # 10-18: place -> coordinates kept (saves a round trip)
                _GEO[gk] = _get("https://geocoding-api.open-meteo.com/v1/search?count=1&name="
                                + urllib.parse.quote(loc.split(",")[0]))["results"][0]
            g = _GEO[gk]
            w = _get(f"https://api.open-meteo.com/v1/forecast?latitude={g['latitude']}&longitude={g['longitude']}"
                     f"&current=temperature_2m,weather_code,wind_speed_10m&temperature_unit={unit}"
                     f"&wind_speed_unit={'mph' if unit == 'fahrenheit' else 'kmh'}")["current"]
            # 10-18: country named (the reader can see WHICH London) and wind in mph with Fahrenheit (no mixed units)
            where = g["name"] + (f", {g['country']}" if g.get("country") else "")
            text = (f"{where}: {round(w['temperature_2m'])} degrees {unit.capitalize()}, "
                    f"{_WMO.get(w['weather_code'], 'unknown conditions')}, wind {round(w['wind_speed_10m'])} "
                    f"{'mph' if unit == 'fahrenheit' else 'km/h'}")
            cache[key] = [text, time.strftime("%b %d %H:%M")]; _save(CACHE, cache)
            _RECENT[key] = (time.time(), dict(ok=True, text=text, source="live"))
            return dict(ok=True, text=text, source="live")
        except Exception:
            pass
    if key in cache:
        return dict(ok=True, text=f"{cache[key][0]} (saved reading from {cache[key][1]})", source="offline")
    return dict(ok=False, text=f"no weather data for {loc} without a network connection", source="offline")


# ------------------------------------------------------------------- translate
_LANG = {"spanish": "es", "french": "fr", "german": "de", "italian": "it", "portuguese": "pt"}
_MT = {}


def translate(text, target_lang):
    code = _LANG.get(str(target_lang).strip().lower(), str(target_lang).strip().lower())
    if code not in _LANG.values():
        return dict(ok=False, text=f"no offline translator for {target_lang}", source="offline")
    try:
        if code not in _MT:
            from transformers import MarianMTModel, MarianTokenizer
            n = f"Helsinki-NLP/opus-mt-en-{code}" if code != "pt" else "Helsinki-NLP/opus-mt-tc-big-en-pt"
            _MT[code] = (MarianTokenizer.from_pretrained(n), MarianMTModel.from_pretrained(n).eval())
        tok, m = _MT[code]
        out = m.generate(**tok([str(text)], return_tensors="pt", truncation=True, max_length=128), max_new_tokens=128)
        return dict(ok=True, text=tok.decode(out[0], skip_special_tokens=True), source="offline")
    except Exception as e:
        return dict(ok=False, text=f"the offline translator for {target_lang} is not installed", source="offline",
                    error=f"{type(e).__name__}: {e}"[:200])


# ------------------------------------------------------------------- simulated
def _outbox(kind, item):
    box = _load(OUTBOX, {"reminders": [], "messages": []})
    box[kind].append(dict(item, created=time.strftime("%Y-%m-%d %H:%M")))
    _save(OUTBOX, box)


def set_reminder(text, datetime):
    _outbox("reminders", {"text": str(text), "when": str(datetime)})
    return dict(ok=True, text=f"reminder saved: {text} ({datetime})", source="simulated")


def send_message(recipient, body):
    _outbox("messages", {"to": str(recipient), "body": str(body)})
    return dict(ok=True, text=f"message to {recipient} placed in the outbox: {body}", source="simulated")


TOOLS = {"calculator": calculator, "unit_convert": unit_convert, "get_time": get_time, "define": define,
         "search": search, "get_weather": get_weather, "translate": translate, "set_reminder": set_reminder,
         "send_message": send_message}
_NET = ("search", "get_weather", "get_time")


def run(name, arguments, offline=False, question=None):
    """Never raises. Unknown tool / bad arguments come back as ok=False with a readable text. `question` = the user's
    message (search uses it to pick the sentence that answers it; the model's query may drop "what time")."""
    f = TOOLS.get(name)
    if f is None:
        return dict(ok=False, text=f"unknown tool: {name}", source="local")
    try:
        kw = dict(arguments or {})
        if name in _NET:
            kw["offline"] = offline
        if name == "search" and question:
            kw["question"] = question
        return f(**kw)
    except TypeError:
        return dict(ok=False, text=f"wrong arguments for {name}", source="local")
    except Exception as e:
        return dict(ok=False, text=f"{name} failed: {type(e).__name__}", source="local")
