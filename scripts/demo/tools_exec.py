"""demo/tools_exec.py -- 2026-10-12 (Neuro 57, docs/demo_v4_plan_2026_10_12.md D2). Executors for the 9 tools of
training_data/posttrain/tools.json. One entry point:

    run(name, arguments, offline=False) -> {"ok": bool, "text": str, "source": "local|live|offline|simulated"}

`text` is the short plain-text result the model reads in its second pass (### Input: Result from <tool>: <text>).
Every network call has a 4 s timeout and falls back to the offline path; nothing here can raise to the caller.
  local      calculator (ast evaluator, never eval), unit_convert (own table), get_time (zoneinfo)
  live       search (Wikipedia REST), get_weather (Open-Meteo, no key); both cached to logs/demo_cache.json
  offline    search (SQLite FTS5 pack data/demo/search_pack.db), define (WordNet via nltk), translate (opus-mt)
  simulated  set_reminder, send_message (stored in logs/demo_outbox.json; nothing leaves the machine)"""
import ast, json, math, operator, os, re, sqlite3, time, datetime, urllib.parse, urllib.request

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
    req = urllib.request.Request(url, headers={"User-Agent": "neuro-demo/0.1 (local research demo)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


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


def get_time(timezone):
    from zoneinfo import ZoneInfo, available_timezones
    q = str(timezone).strip()
    name = None
    if q in available_timezones():
        name = q
    else:
        k = q.lower().replace("_", " ").split(",")[0].strip()
        name = _TZ.get(k)
        if name is None:                      # "Sao_Paulo" -> America/Sao_Paulo
            hit = [z for z in available_timezones() if z.lower().replace("_", " ").endswith("/" + k)]
            name = sorted(hit)[0] if hit else None
    if name is None:
        return dict(ok=False, text=f"unknown timezone: {timezone}", source="local")
    now = datetime.datetime.now(ZoneInfo(name))
    return dict(ok=True, text=f"{now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B')} {now.day} ({name})",
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


def search(query, offline=False):
    q = str(query).strip()
    cache = _load(CACHE, {})
    key = "search|" + q.lower()
    if not offline:
        try:
            hits = _get("https://en.wikipedia.org/w/rest.php/v1/search/page?limit=1&q=" + urllib.parse.quote(q))
            title = hits["pages"][0]["key"]
            s = _get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title, safe=""))
            if s.get("extract"):
                res = dict(ok=True, text=_clip(f"{s.get('title', title)}: {s['extract']}"), source="live")
                cache[key] = res["text"]; _save(CACHE, cache)
                return res
        except Exception:
            pass
    if key in cache:
        return dict(ok=True, text=cache[key], source="offline")
    return search_offline(q)


# ----------------------------------------------------------------- get_weather
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle",
        53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
        73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def get_weather(location, unit="celsius", offline=False):
    loc = str(location).strip()
    unit = "fahrenheit" if str(unit).lower().startswith("f") else "celsius"
    cache = _load(CACHE, {})
    key = f"weather|{loc.lower()}|{unit}"
    if not offline:
        try:
            g = _get("https://geocoding-api.open-meteo.com/v1/search?count=1&name="
                     + urllib.parse.quote(loc.split(",")[0]))["results"][0]
            w = _get(f"https://api.open-meteo.com/v1/forecast?latitude={g['latitude']}&longitude={g['longitude']}"
                     f"&current=temperature_2m,weather_code,wind_speed_10m&temperature_unit={unit}")["current"]
            text = (f"{g['name']}: {round(w['temperature_2m'])} degrees {unit.capitalize()}, "
                    f"{_WMO.get(w['weather_code'], 'unknown conditions')}, wind {round(w['wind_speed_10m'])} km/h")
            cache[key] = [text, time.strftime("%b %d %H:%M")]; _save(CACHE, cache)
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
_NET = ("search", "get_weather")


def run(name, arguments, offline=False):
    """Never raises. Unknown tool / bad arguments come back as ok=False with a readable text."""
    f = TOOLS.get(name)
    if f is None:
        return dict(ok=False, text=f"unknown tool: {name}", source="local")
    try:
        kw = dict(arguments or {})
        if name in _NET:
            kw["offline"] = offline
        return f(**kw)
    except TypeError:
        return dict(ok=False, text=f"wrong arguments for {name}", source="local")
    except Exception as e:
        return dict(ok=False, text=f"{name} failed: {type(e).__name__}", source="local")
