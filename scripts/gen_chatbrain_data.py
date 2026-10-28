"""gen_chatbrain_data.py -- 2026-10-21 (Neuro 72). Chat brain step 3 (docs/chat_brain_plan_2026_10_21.md): multi-turn
SFT data for a LoRA brain on sft_v4_final_slim. CPU only; reads the teacher lists of temp/n72_chat_teacher.py.

A chat is a list of turns (user message, the reply the user saw, training target or None). Every turn with a target
becomes one record whose history is the turns before it ("sliding target": loss on that turn only). The history is
rendered by sft_data.render_conversation, the same function the demo engine uses.

  study.jsonl   decide-pass records (tool format: target_mode / target_call / target_text, + "history"), all 9 tools
                offered, plus 1:1 single-turn replay (tools.jsonl + selfknow_cF, train split only)
  answer.jsonl  answer-pass records (instruction / input / output, + "history" = the last turn), plus 1:1 replay of
                toolresult_blend without history
  quiz.jsonl    held-out chats (other names / cities / words / phrases / entities; tools.jsonl TEST split for topic
                switches): {"cat", "turns": [user messages], "check": {...}} for temp/n72_chat_quiz.py

  python scripts/gen_chatbrain_data.py   -> training_data/brains/chat/"""
import os, sys, json, random, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(R, "src"), os.path.join(R, "scripts", "demo")]
import tool_data as td
import tools_exec as TX

OUT = os.path.join(R, "training_data", "brains", "chat")
P = os.path.join(R, "training_data", "posttrain")
rng = random.Random(721)
TOOLS = td.load_tools()
ALL = [t["name"] for t in TOOLS]
pick = rng.choice

# ------------------------------------------------------------------------------------------------ slot lists (train / quiz)
NAMES = (["Maria", "James", "Aisha", "Lucas", "Sofia", "Daniel", "Mei", "Omar", "Elena", "Noah", "Priya", "Liam", "Fatima",
          "Ethan", "Yuki", "Mateo", "Hannah", "Kofi", "Chloe", "Ivan", "Nina", "Samuel", "Leila", "Oscar", "Grace", "Raj",
          "Emma", "Diego", "Zara", "Peter", "Anya", "Marcus", "Ines", "Tom", "Rosa", "Kenji", "Laura", "Ali", "Julia",
          "Felix", "Amara", "Victor", "Clara", "Hugo", "Sara", "Ben", "Olga", "Luis", "Mia", "Arjun"],
         ["Beatrice", "Tariq", "Greta", "Rafael", "Ingrid", "Musa", "Freya", "Dmitri", "Paloma", "Soren", "Nadia", "Wesley",
          "Imani", "Bruno", "Ayla"])
CITIES = ({"Paris": "Europe/Paris", "London": "Europe/London", "Berlin": "Europe/Berlin", "Madrid": "Europe/Madrid",
           "Rome": "Europe/Rome", "Tokyo": "Asia/Tokyo", "Seoul": "Asia/Seoul", "Beijing": "Asia/Shanghai",
           "Sydney": "Australia/Sydney", "Toronto": "America/Toronto", "Chicago": "America/Chicago",
           "New York": "America/New_York", "Los Angeles": "America/Los_Angeles", "Denver": "America/Denver",
           "Mexico City": "America/Mexico_City", "Cairo": "Africa/Cairo", "Nairobi": "Africa/Nairobi", "Lagos": "Africa/Lagos",
           "Mumbai": "Asia/Kolkata", "Dubai": "Asia/Dubai", "Moscow": "Europe/Moscow", "Istanbul": "Europe/Istanbul",
           "Athens": "Europe/Athens", "Lisbon": "Europe/Lisbon", "Dublin": "Europe/Dublin", "Amsterdam": "Europe/Amsterdam",
           "Stockholm": "Europe/Stockholm", "Oslo": "Europe/Oslo", "Vienna": "Europe/Vienna", "Prague": "Europe/Prague",
           "Warsaw": "Europe/Warsaw", "Bangkok": "Asia/Bangkok", "Singapore": "Asia/Singapore", "Jakarta": "Asia/Jakarta",
           "Manila": "Asia/Manila", "Hong Kong": "Asia/Hong_Kong", "Buenos Aires": "America/Argentina/Buenos_Aires",
           "Sao Paulo": "America/Sao_Paulo", "Lima": "America/Lima", "Bogota": "America/Bogota",
           "Santiago": "America/Santiago", "Vancouver": "America/Vancouver", "Honolulu": "Pacific/Honolulu",
           "Auckland": "Pacific/Auckland", "Johannesburg": "Africa/Johannesburg", "Karachi": "Asia/Karachi"},
          {"Helsinki": "Europe/Helsinki", "Budapest": "Europe/Budapest", "Zurich": "Europe/Zurich",
           "Brussels": "Europe/Brussels", "Copenhagen": "Europe/Copenhagen", "Kuala Lumpur": "Asia/Kuala_Lumpur",
           "Taipei": "Asia/Taipei", "Phoenix": "America/Phoenix", "Havana": "America/Havana", "Accra": "Africa/Accra",
           "Riyadh": "Asia/Riyadh", "Perth": "Australia/Perth"})
JOBS = (["teacher", "nurse", "software engineer", "chef", "carpenter", "lawyer", "pilot", "farmer", "dentist", "accountant",
         "photographer", "electrician", "librarian", "firefighter", "architect", "mechanic", "journalist", "pharmacist"],
        ["veterinarian", "plumber", "translator", "baker", "geologist", "tailor"])
PETS = ([("dog", n) for n in ("Max", "Bella", "Rocky", "Luna", "Buddy", "Daisy", "Charlie", "Rex")] +
        [("cat", n) for n in ("Whiskers", "Milo", "Shadow", "Cleo", "Simba", "Pepper", "Oliver", "Nala")] +
        [("parrot", n) for n in ("Kiwi", "Polly", "Mango")] + [("rabbit", n) for n in ("Thumper", "Coco", "Hazel")],
        [("dog", n) for n in ("Biscuit", "Ziggy")] + [("cat", n) for n in ("Mittens", "Juniper")] +
        [("hamster", n) for n in ("Peanut", "Nibbles")])
RELS = (["sister", "brother", "mother", "father", "wife", "husband", "son", "daughter", "best friend", "boss"],
        ["grandmother", "cousin", "neighbor", "uncle"])
FAVS = ([("color", v) for v in ("green", "blue", "red", "purple", "yellow", "orange")] +
        [("food", v) for v in ("pizza", "sushi", "tacos", "lasagna", "pancakes", "curry")] +
        [("sport", v) for v in ("tennis", "soccer", "basketball", "swimming", "golf")] +
        [("season", v) for v in ("summer", "winter", "spring", "autumn")] +
        [("animal", v) for v in ("the owl", "the dolphin", "the tiger", "the horse")],
        [("color", "teal"), ("color", "silver"), ("food", "paella"), ("food", "dumplings"), ("sport", "volleyball"),
         ("animal", "the fox")])
YEARS = ([y for y in range(1950, 2008) if y % 4 != 1], [y for y in range(1950, 2008) if y % 4 == 1])
WORDS = (["serendipity", "ephemeral", "ubiquitous", "meticulous", "resilient", "candid", "frugal", "lucid", "pragmatic",
          "tenacious", "verbose", "benevolent", "nostalgia", "eloquent", "diligent", "ambiguous", "gregarious", "humble",
          "obsolete", "vivid", "arbitrary", "cynical", "elusive", "fickle", "gullible", "lethargic", "mundane", "placid"],
         ["melancholy", "audacious", "whimsical", "stoic", "zealous", "aloof", "jovial", "sporadic"])
PHRASES = ({"good morning": ["Buenos días", "Bonjour", "Guten Morgen", "Buongiorno", "Bom dia"],
            "good night": ["Buenas noches", "Bonne nuit", "Gute Nacht", "Buonanotte", "Boa noite"],
            "thank you": ["Gracias", "Merci", "Danke", "Grazie", "Obrigado"],
            "how are you": ["¿Cómo estás?", "Comment ça va ?", "Wie geht es dir?", "Come stai?", "Como vai você?"],
            "see you tomorrow": ["Hasta mañana", "À demain", "Bis morgen", "A domani", "Até amanhã"],
            "I love you": ["Te quiero", "Je t'aime", "Ich liebe dich", "Ti amo", "Eu te amo"],
            "where is the bathroom": ["¿Dónde está el baño?", "Où sont les toilettes ?", "Wo ist die Toilette?",
                                      "Dov'è il bagno?", "Onde fica o banheiro?"],
            "happy birthday": ["Feliz cumpleaños", "Joyeux anniversaire", "Alles Gute zum Geburtstag", "Buon compleanno",
                               "Feliz aniversário"],
            "excuse me": ["Disculpe", "Excusez-moi", "Entschuldigung", "Mi scusi", "Com licença"],
            "I am hungry": ["Tengo hambre", "J'ai faim", "Ich habe Hunger", "Ho fame", "Estou com fome"],
            "what time is it": ["¿Qué hora es?", "Quelle heure est-il ?", "Wie spät ist es?", "Che ore sono?", "Que horas são?"],
            "how much does it cost": ["¿Cuánto cuesta?", "Combien ça coûte ?", "Wie viel kostet das?", "Quanto costa?",
                                      "Quanto custa?"],
            "I don't understand": ["No entiendo", "Je ne comprends pas", "Ich verstehe nicht", "Non capisco", "Eu não entendo"],
            "please": ["Por favor", "S'il vous plaît", "Bitte", "Per favore", "Por favor"],
            "goodbye": ["Adiós", "Au revoir", "Auf Wiedersehen", "Arrivederci", "Adeus"],
            "the bill, please": ["La cuenta, por favor", "L'addition, s'il vous plaît", "Die Rechnung, bitte",
                                 "Il conto, per favore", "A conta, por favor"],
            "welcome": ["Bienvenido", "Bienvenue", "Willkommen", "Benvenuto", "Bem-vindo"],
            "good luck": ["Buena suerte", "Bonne chance", "Viel Glück", "Buona fortuna", "Boa sorte"]},
           {"congratulations": ["Felicidades", "Félicitations", "Herzlichen Glückwunsch", "Congratulazioni", "Parabéns"],
            "see you later": ["Hasta luego", "À plus tard", "Bis später", "A più tardi", "Até logo"],
            "I need help": ["Necesito ayuda", "J'ai besoin d'aide", "Ich brauche Hilfe", "Ho bisogno di aiuto",
                            "Preciso de ajuda"],
            "have a nice day": ["Que tengas un buen día", "Bonne journée", "Schönen Tag noch", "Buona giornata",
                                "Tenha um bom dia"]})
LANGS = ["Spanish", "French", "German", "Italian", "Portuguese"]
UNITS = {"length": ["kilometers", "miles", "meters", "feet"], "weight": ["kilograms", "pounds", "grams", "ounces"],
         "volume": ["liters", "gallons"], "temperature": ["celsius", "fahrenheit"]}
TASKS = (["call mom", "water the plants", "pay the rent", "buy milk", "take my medicine", "walk the dog", "send the report",
          "book a dentist appointment", "pick up the kids", "renew my passport", "call the bank", "charge my phone"],
         ["feed the cat", "return the library books", "check the oven", "email my teacher"])
WHENS = ["at 5 pm", "at 7 am", "at noon", "tomorrow at 9 am", "tonight at 8 pm", "on Friday at 3 pm", "in 30 minutes",
         "in an hour", "on Monday morning", "at 6:30 pm"]
BODIES = ["I will be late", "dinner is ready", "the meeting moved to 3 pm", "I'm on my way", "happy birthday",
          "call me when you can", "see you at the station", "thanks for your help"]
CONDS = ["clear sky", "partly cloudy", "overcast", "light rain", "sunny", "foggy", "light snow", "thunderstorms"]


def S(pair, quiz):
    return pair[1] if quiz else pair[0]


def an(w):
    return ("an " if w[0] in "aeiou" else "a ") + w


# ------------------------------------------------------------------------------------------------ the chat model
def T(user, reply, target=None):
    """target: ("ANSWER", text) | ("TOOL", {"name", "arguments"}) | None (history only)."""
    return {"user": user, "reply": reply, "target": target}


def A(user, text):
    return T(user, text, ("ANSWER", text))


def num(x):
    x = float(x)
    return int(x) if x == int(x) else round(x, 4)


def exec_text(name, args):
    r = TX.run(name, args, offline=True)
    return r["text"] if r.get("ok") else None


# distractors: realistic earlier turns (question + the reply the demo gave), history only
TR = [json.loads(l) for l in open(os.path.join(P, "toolresult_blend.jsonl"))]
TR_TEST = {json.loads(l)["instruction"] for f in ("toolresult_test.jsonl", "toolresult_search_test.jsonl")
           for l in open(os.path.join(P, f))}
TR = [r for r in TR if r["instruction"] not in TR_TEST and len(r["output"].split()) <= 50]
rng.shuffle(TR)
DISTRACT = [T(r["instruction"], r["output"]) for r in TR[:3000]]
ANOTHER = [json.loads(l) for l in open(os.path.join(OUT, "another.jsonl"))]
random.Random(7).shuffle(ANOTHER)
ANOTHER_Q, ANOTHER = ANOTHER[:30], ANOTHER[30:]          # the quiz's lists are not trained


def distract(k):
    return [dict(pick(DISTRACT)) for _ in range(k)]


# ------------------------------------------------------------------------------------------------ facts + honesty + recall
def facts(quiz):
    nm, city, job = pick(S(NAMES, quiz)), pick(list(S(CITIES, quiz))), pick(S(JOBS, quiz))
    animal, pet = pick(S(PETS, quiz)); rel, rn = pick(S(RELS, quiz)), pick(S(NAMES, quiz))
    thing, fav = pick(S(FAVS, quiz)); year = pick(S(YEARS, quiz))
    F = {"name": (pick([f"My name is {nm}.", f"I'm {nm}.", f"Hi, I'm {nm}.", f"You can call me {nm}."]),
                  pick([f"Nice to meet you, {nm}.", f"Hello, {nm}! How can I help you?"]),
                  pick(["What is my name?", "What's my name?", "Do you remember my name?", "Who am I?",
                        "What did I say my name was?"]),
                  pick([f"Your name is {nm}.", f"You told me your name is {nm}."]), nm,
                  "You haven't told me your name."),
         "city": (pick([f"I live in {city}.", f"I just moved to {city}.", f"Home for me is {city}."]),
                  pick([f"Got it, you live in {city}.", f"{city}, nice! How can I help?"]),
                  pick(["Where do I live?", "Which city do I live in?", "Do you remember where I live?"]),
                  f"You live in {city}.", city, "You haven't told me where you live."),
         "job": (pick([f"I work as {an(job)}.", f"I'm {an(job)} by profession."]),
                 pick([f"Got it, you work as {an(job)}.", f"That sounds interesting! How can I help?"]),
                 pick(["What do I do for a living?", "What is my job?", "What's my job?"]),
                 f"You work as {an(job)}.", job, "You haven't told me what you do for work."),
         "pet": (pick([f"I have {an(animal)} named {pet}.", f"My {animal} is called {pet}."]),
                 pick([f"{pet} is a lovely name for {an(animal)}!", f"Got it, your {animal} is called {pet}."]),
                 pick([f"What is my {animal}'s name?", f"What's my {animal} called?",
                       f"What did I say my {animal} is called?"]),
                 f"Your {animal} is called {pet}.", pet, f"You haven't told me your {animal}'s name."),
         "rel": (pick([f"My {rel}'s name is {rn}.", f"My {rel} is called {rn}."]),
                 pick([f"Got it, your {rel} is {rn}.", f"{rn}, that's a nice name."]),
                 pick([f"What is my {rel}'s name?", f"What's my {rel} called?"]),
                 f"Your {rel}'s name is {rn}.", rn, f"You haven't told me your {rel}'s name."),
         "born": (f"I was born in {year}.", f"Got it, you were born in {year}.",
                  pick(["What year was I born?", "When was I born?", "In which year was I born?"]),
                  f"You were born in {year}.", str(year), "You haven't told me when you were born."),
         "fav": (pick([f"My favorite {thing} is {fav}.", f"I love {fav}, it's my favorite {thing}."]),
                 pick([f"{fav[0].upper() + fav[1:]}, great choice!", f"Got it, your favorite {thing} is {fav}."]),
                 pick([f"What is my favorite {thing}?", f"What's my favorite {thing}?",
                       f"Do you remember my favorite {thing}?"]),
                 f"Your favorite {thing} is {fav}.", fav.replace("the ", ""), f"You haven't told me your favorite {thing}.")}
    return F


def chat_fact(quiz):
    F = facts(quiz); ks = rng.sample(list(F), rng.choice([1, 1, 2]))
    turns = []
    for k in ks:
        turns += distract(rng.choice([0, 0, 1])) + [A(F[k][0], F[k][1])]
    turns += distract(rng.choice([0, 1, 1, 2]))
    k = pick(ks)
    turns.append(A(F[k][2], F[k][3]))
    return turns, {"reply": F[k][4]}


def chat_honest(quiz):
    F = facts(quiz); given = rng.sample(list(F), rng.choice([0, 1, 1, 2]))
    turns = []
    for k in given:
        turns += [A(F[k][0], F[k][1])] + distract(rng.choice([0, 1]))
    turns += distract(rng.choice([0, 1]) if given else rng.choice([0, 0, 1, 2]))
    k = pick([x for x in F if x not in given])
    # a pet / relation question about one the user never mentioned: the given one has another animal / relation
    turns.append(A(F[k][2], F[k][5]))
    return turns, {"reply": "haven't told"}


def chat_recall(quiz):
    turns = distract(rng.choice([1, 2, 3]))
    if len(turns) >= 2 and rng.random() < 0.4:
        turns.append(A(pick(["What was my first question?", "What did I ask you first?"]),
                       f"Your first question was: \"{turns[0]['user']}\""))
        return turns, {"reply": turns[0]["user"][:25]}
    prev = turns[-1]["user"]
    turns.append(A(pick(["What did I just ask you?", "What was my last question?", "What did I ask you before?"]),
                   f"You asked: \"{prev}\""))
    return turns, {"reply": prev[:25]}


def chat_recall_none():
    return [A(pick(["What did I ask you before?", "What was my last question?"]),
              "You haven't asked me anything before this.")], {"reply": "haven't asked"}


# ------------------------------------------------------------------------------------------------ another one
FIRST = {"joke": ["Tell me a joke about {t}.", "Do you know a joke about {t}?", "Tell me a funny joke about {t}.",
                  "I'd like a joke about {t}."],
         "tip": ["Give me a tip for {t}.", "Any advice for {t}?", "What's a good tip for {t}?"],
         "idea": ["I need {t}. Can you suggest one?", "Give me one idea: {t}.", "Can you suggest one of the {t}?"],
         "syn": ["What's another word for \"{t}\"?", "Give me a synonym for {t}.", "What is a synonym of {t}?"]}
MORE = {"joke": ["Tell me another one.", "Another one, please.", "Give me another one.", "One more?",
                 "Can you tell me a different one?", "Ha! Another one?", "Got any more?", "Tell me one more."],
        "tip": ["Another one, please.", "Got another tip?", "Give me another one.", "What else?", "One more tip, please."],
        "idea": ["Another one, please.", "Any other ideas?", "Give me another one.", "Something different?", "One more?"],
        "syn": ["Another one?", "Give me a different word.", "Any other synonyms?", "One more?"]}


def say_item(kind, t, item, first):
    if kind == "syn":
        return f"Another word for {t} is \"{item}\"." if first else f"You could also say \"{item}\"."
    return item


def chat_another(row):
    kind, t, items = row["kind"], row["topic"], list(row["items"])
    rng.shuffle(items)
    turns = distract(rng.choice([0, 0, 1])) + [A(pick(FIRST[kind]).format(t=t), say_item(kind, t, items[0], True))]
    for i in range(1, rng.choice([2, 2, 3]) if len(items) >= 3 else 2):
        turns.append(A(pick(MORE[kind]), say_item(kind, t, items[i], False)))
    return turns, {"differs": True}


# ------------------------------------------------------------------------------------------------ tool follow-ups
def call(name, **a):
    return {"name": name, "arguments": a}


ANS = []          # answer-pass records (instruction, input, output, history)


def tool_turn(user, c, result, reply, history):
    ANS.append({"instruction": user, "input": f"Result from {c['name']}: {result}", "output": reply,
                "history": [[t["user"], t["reply"]] for t in history[-1:]]})
    return T(user, reply, ("TOOL", c))


def weather_res(city, unit):
    deg = rng.randint(-5, 35) if unit == "celsius" else rng.randint(20, 98)
    cond = pick(CONDS)
    return f"{city}: {deg} degrees {unit.capitalize()}, {cond}, wind {rng.randint(2, 30)} km/h", \
        f"It is {deg} degrees {unit.capitalize()} in {city}, {cond}."


def chat_weather(quiz):
    c1, c2 = rng.sample(list(S(CITIES, quiz)), 2)
    u = "celsius" if rng.random() < 0.3 else "fahrenheit"
    q1 = pick([f"What's the weather in {c1}?", f"How's the weather in {c1} today?", f"What is the weather like in {c1}?"])
    if u == "celsius":
        q1 = q1.rstrip("?") + " in celsius?"
    h = distract(rng.choice([0, 0, 1]))
    res, rep = weather_res(c1, u)
    h.append(tool_turn(q1, call("get_weather", location=c1, unit=u), res, rep, h))
    r = rng.random()
    if r < 0.7:
        q2 = pick([f"What about {c2}?", f"And in {c2}?", f"How about {c2}?", f"And {c2}?", f"Sorry, I meant {c2}.",
                   f"What's it like in {c2}?", f"Same for {c2}, please."])
        c, chk = call("get_weather", location=c2, unit=u), {"tool": "get_weather", "args": [c2]}
        res, rep = weather_res(c2, u)
    elif r < 0.85:
        u2 = "fahrenheit" if u == "celsius" else "celsius"
        q2 = pick([f"Can you give me that in {u2}?", f"In {u2}, please.", f"And in {u2}?"])
        c, chk = call("get_weather", location=c1, unit=u2), {"tool": "get_weather", "args": [c1, u2]}
        res, rep = weather_res(c1, u2)
    else:
        q2 = pick(["Is it raining there?", "Will it be windy there?", "Do I need an umbrella there?"])
        c, chk = call("get_weather", location=c1, unit=u), {"tool": "get_weather", "args": [c1]}
        res, rep = weather_res(c1, u)
    h.append(tool_turn(q2, c, res, rep, h))
    return h, chk


def time_res(tz, city):
    t = exec_text("get_time", {"timezone": tz})
    return t, f"It is {t.rsplit(' (', 1)[0]} in {city}."


def chat_time(quiz):
    cs = S(CITIES, quiz); c1, c2 = rng.sample(list(cs), 2)
    q1 = pick([f"What time is it in {c1}?", f"What's the time in {c1} right now?", f"Can you tell me the time in {c1}?"])
    h = distract(rng.choice([0, 0, 1]))
    res, rep = time_res(cs[c1], c1)
    h.append(tool_turn(q1, call("get_time", timezone=cs[c1]), res, rep, h))
    q2 = pick([f"And in {c2}?", f"What about {c2}?", f"How about {c2}?", f"And {c2}?", f"Sorry, I meant {c2}."])
    res, rep = time_res(cs[c2], c2)
    h.append(tool_turn(q2, call("get_time", timezone=cs[c2]), res, rep, h))
    return h, {"tool": "get_time", "args": [c2.split()[0]]}       # "Kuala" in "Asia/Kuala_Lumpur" and "Kuala Lumpur"


def conv_res(v, f, t):
    x = exec_text("unit_convert", {"value": v, "from_unit": f, "to_unit": t})
    if not x or "=" not in x:
        return None, None, None
    r = x.split("=", 1)[1].strip().split(" ")[0]
    return x, f"{v} {f} is {r} {t}.", r


def chat_convert(quiz):
    dim = pick(list(UNITS)); f, t = rng.sample(UNITS[dim], 2)
    v = num(pick([rng.randint(2, 500), round(rng.uniform(1, 300), 1)]))
    q1 = pick([f"How many {t} is {v} {f}?", f"Convert {v} {f} to {t}.", f"What is {v} {f} in {t}?"])
    h = distract(rng.choice([0, 0, 1]))
    res, rep, r = conv_res(v, f, t)
    if res is None:
        return None
    h.append(tool_turn(q1, call("unit_convert", value=v, from_unit=f, to_unit=t), res, rep, h))
    others = [u for u in UNITS[dim] if u not in (f, t)]
    if others and rng.random() < 0.35:
        t2 = pick(others); v2 = num(r.replace(",", ""))
        q2 = pick([f"Convert that to {t2}.", f"And how much is that in {t2}?", f"Now in {t2}, please."])
        c, (res, rep, _) = call("unit_convert", value=v2, from_unit=t, to_unit=t2), conv_res(v2, t, t2)
        chk = {"tool": "unit_convert", "args": [t2]}
    else:
        v2 = num(pick([rng.randint(2, 500), round(rng.uniform(1, 300), 1)]))
        q2 = pick([f"And {v2} {f}?", f"What about {v2}?", f"And {v2}?", f"Now {v2} {f}, please.", f"Sorry, I meant {v2} {f}."])
        c, (res, rep, _) = call("unit_convert", value=v2, from_unit=f, to_unit=t), conv_res(v2, f, t)
        chk = {"tool": "unit_convert", "args": [str(v2)]}
    if res is None:
        return None
    h.append(tool_turn(q2, c, res, rep, h))
    return h, chk


def calc_res(expr):
    return exec_text("calculator", {"expression": expr})


def chat_calc(quiz):
    h = distract(rng.choice([0, 0, 1]))
    if rng.random() < 0.5:
        a, b = rng.choice([5, 10, 12, 15, 18, 20, 25, 30, 40, 75]), rng.choice([40, 60, 80, 120, 150, 240, 360, 500, 1200])
        e1 = f"{a / 100:g} * {b}"; r1 = calc_res(e1)
        h.append(tool_turn(pick([f"What is {a}% of {b}?", f"What's {a} percent of {b}?"]), call("calculator", expression=e1),
                           r1, f"{a}% of {b} is {r1}.", h))
        if rng.random() < 0.5:
            c2 = rng.choice([x for x in (5, 10, 15, 20, 25, 30, 50) if x != a]); e2 = f"{c2 / 100:g} * {b}"
            q2 = pick([f"And {c2}%?", f"What about {c2}%?", f"And {c2} percent?"]); rep = f"{c2}% of {b} is {calc_res(e2)}."
            chk = {"tool": "calculator", "args": [str(b)]}
        else:
            d = rng.choice([x for x in (50, 90, 200, 300, 640, 800) if x != b]); e2 = f"{a / 100:g} * {d}"
            q2 = pick([f"And of {d}?", f"What about {a}% of {d}?", f"And for {d}?"]); rep = f"{a}% of {d} is {calc_res(e2)}."
            chk = {"tool": "calculator", "args": [str(d)]}
    else:
        x, y = rng.randint(12, 999), rng.randint(2, 99)
        op, w = pick([("*", "times"), ("+", "plus"), ("-", "minus")])
        e1 = f"{x} {op} {y}"; r1 = calc_res(e1)
        h.append(tool_turn(pick([f"What is {x} {w} {y}?", f"Calculate {x} {w} {y}."]), call("calculator", expression=e1),
                           r1, f"{x} {w} {y} is {r1}.", h))
        z = rng.randint(2, 12); op2, w2 = pick([("*", "Multiply that by"), ("+", "Add"), ("/", "Divide that by")])
        q2 = f"{w2} {z}." if op2 != "+" else f"Now add {z} to that."
        e2 = f"{r1.replace(',', '')} {op2} {z}"; rep = f"That gives {calc_res(e2)}."
        chk = {"tool": "calculator", "args": [str(z)]}
    if calc_res(e2) is None:
        return None
    h.append(tool_turn(q2, call("calculator", expression=e2), calc_res(e2), rep, h))
    return h, chk


def chat_translate(quiz):
    ph = S(PHRASES, quiz); p = pick(list(ph)); l1, l2 = rng.sample(range(5), 2)
    q1 = pick([f"Translate '{p}' into {LANGS[l1]}.", f"How do you say '{p}' in {LANGS[l1]}?", f"What is '{p}' in {LANGS[l1]}?"])
    h = distract(rng.choice([0, 0, 1]))
    h.append(tool_turn(q1, call("translate", text=p, target_lang=LANGS[l1]), ph[p][l1], ph[p][l1], h))
    q2 = pick([f"Now into {LANGS[l2]}.", f"And in {LANGS[l2]}?", f"What about {LANGS[l2]}?", f"How about in {LANGS[l2]}?",
               f"Sorry, I meant {LANGS[l2]}."])
    h.append(tool_turn(q2, call("translate", text=p, target_lang=LANGS[l2]), ph[p][l2], ph[p][l2], h))
    return h, {"tool": "translate", "args": [LANGS[l2], p.split()[0]]}


def define_res(w):
    t = exec_text("define", {"term": w})
    if not t or "):" not in t:
        return None, None
    return t, f"{w[0].upper() + w[1:]} means {t.split('):', 1)[1].strip().rstrip('.')}."


def chat_define(quiz):
    w1, w2 = rng.sample(S(WORDS, quiz), 2)
    r1, a1 = define_res(w1); r2, a2 = define_res(w2)
    if not (r1 and r2):
        return None
    h = distract(rng.choice([0, 0, 1]))
    h.append(tool_turn(pick([f"Define {w1}.", f"What does {w1} mean?", f"What is the meaning of {w1}?"]),
                       call("define", term=w1), r1, a1, h))
    h.append(tool_turn(pick([f"What about {w2}?", f"And {w2}?", f"How about {w2}?"]), call("define", term=w2), r2, a2, h))
    return h, {"tool": "define", "args": [w2]}


def chat_remind(quiz):
    task = pick(S(TASKS, quiz)); w1, w2 = rng.sample(WHENS, 2)
    text = re.sub(r"\bmy\b", "your", task)
    h = distract(rng.choice([0, 0, 1]))
    h.append(tool_turn(f"Remind me to {task} {w1}.", call("set_reminder", text=text, datetime=w1),
                       f"reminder saved: {text} ({w1})", f"OK, I set a reminder to {text} {w1}.", h))
    q2 = pick([f"Actually, make it {w2}.", f"Can you change that to {w2}?", f"Sorry, I meant {w2}."])
    h.append(tool_turn(q2, call("set_reminder", text=text, datetime=w2), f"reminder saved: {text} ({w2})",
                       f"OK, I changed the reminder to {text} {w2}.", h))
    key = next((w for w in w2.split() if re.search(r"\d", w)), w2.split()[-1])   # "6:30", "30", "noon", "hour"
    return h, {"tool": "set_reminder", "args": [key]}


def chat_message(quiz):
    n1, n2 = rng.sample(S(NAMES, quiz), 2); body = pick(BODIES)
    h = distract(rng.choice([0, 0, 1]))
    h.append(tool_turn(f"Send a message to {n1} saying {body}.", call("send_message", recipient=n1, body=body),
                       f"message to {n1} placed in the outbox: {body}", f"Your message to {n1} is in the outbox: {body}.", h))
    q2 = pick([f"Send the same message to {n2}.", f"Please send it to {n2} as well.", f"Also send it to {n2}."])
    h.append(tool_turn(q2, call("send_message", recipient=n2, body=body), f"message to {n2} placed in the outbox: {body}",
                       f"Your message to {n2} is in the outbox: {body}.", h))
    return h, {"tool": "send_message", "args": [n2, body.split()[-1]]}


TOOL_CHATS = [chat_weather, chat_time, chat_convert, chat_calc, chat_translate, chat_define, chat_remind, chat_message]


# ------------------------------------------------------------------------------------------------ pronoun follow-ups
PRO = [json.loads(l) for l in open(os.path.join(OUT, "pronoun.jsonl"))] if os.path.exists(os.path.join(OUT, "pronoun.jsonl")) \
    else (print("[warn] no pronoun.jsonl (teacher pass not done): no pronoun chats") or [])
_PW = {"he", "she", "it", "they", "his", "her", "its", "their", "him", "them"}
PRO = [r for r in PRO if not (set(re.findall(r"[a-z]+", r["q2"].lower())) - _PW
                              <= set(re.findall(r"[a-z]+", r["q1"].lower())))]   # "Where was he born?" after the same q


def chat_pronoun(row):
    h = distract(rng.choice([0, 0, 1])) + [T(row["q1"], row["a1"])]
    h.append(T(row["q2"], "", ("TOOL", call("search", query=row["query"]))))
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", row["title"]) if w.lower() not in ("the", "and")]
    return h, {"tool": "search", "args_any": [w for w in words][:3]}


# ------------------------------------------------------------------------------------------------ topic switch
def single(split):
    rows = [json.loads(l) for l in open(os.path.join(P, "tools.jsonl"))] + \
           [json.loads(l) for l in open(os.path.join(P, "selfknow_cF.jsonl"))]
    return [r for r in rows if td.split_of(r["instruction"]) == split]


SINGLE_TR, SINGLE_TE = single("train"), single("test")


def switch_turn(r):
    if r["target_mode"] == "TOOL":
        return T(r["instruction"], "", ("TOOL", r["target_call"])), r
    return T(r["instruction"], r["target_text"], ("ANSWER", r["target_text"])), r


def chat_switch(pool, quiz):
    F = facts(quiz); k = pick(list(F))
    prior = rng.choice([distract(rng.choice([1, 2])), [A(F[k][0], F[k][1])] + distract(rng.choice([0, 1])),
                        chat_another(pick(ANOTHER))[0][-2:]])
    t, r = switch_turn(pick(pool))
    if r["target_mode"] == "TOOL":
        chk = {"tool": r["target_call"]["name"]}
    else:
        chk = {"tool": None}
    return prior + [t], chk, r["tools"]


# ------------------------------------------------------------------------------------------------ build
def records(turns, offered=None, last_only=False):
    out = []
    for i, t in enumerate(turns):
        if t["target"] is None or (last_only and i != len(turns) - 1):
            continue
        hist = [[x["user"], x["reply"] or "(looked it up)"] for x in turns[:i]]
        mode, tgt = t["target"]
        rec = {"kind": "chat", "instruction": t["user"], "history": hist, "tools": offered or ALL, "target_mode": mode}
        rec.update({"target_call": tgt} if mode == "TOOL" else {"target_text": tgt})
        out.append(rec)
    return out


def main():
    global OUT
    OUT = os.environ.get("CHATBRAIN_OUT", OUT)          # a dry run writes elsewhere (inputs are still read from OUT)
    os.makedirs(OUT, exist_ok=True)
    study, quiz, counts = [], [], {}

    def add(cat, recs):
        study.extend(recs); counts[cat] = counts.get(cat, 0) + len(recs)
    for _ in range(1200):
        add("fact", records(chat_fact(False)[0]))
    for _ in range(450):
        add("honest", records(chat_honest(False)[0], last_only=True))
    for _ in range(80):
        add("honest", records(chat_recall_none()[0]))
    for _ in range(300):
        add("recall", records(chat_recall(False)[0], last_only=True))
    for row in ANOTHER:
        for _ in range(6 if row["kind"] != "syn" else 4):
            add("another", records(chat_another(row)[0]))
    for row in [r for r in PRO if r["split"] == "train"]:
        add("pronoun", records(chat_pronoun(row)[0]))
    ANS.clear()
    n_tool = 0
    while n_tool < 1700:
        r = pick(TOOL_CHATS)(False)
        if r:
            add("tool_followup", records(r[0])); n_tool += 1
    ans_chat = list(ANS)
    for _ in range(1400):
        turns, _, offered = chat_switch(SINGLE_TR, False)
        add("switch", records(turns, offered, last_only=True))
    n_chat = len(study)
    # 1:1 single-turn replay (train split; load_tool_jsonl drops the test split anyway)
    rep = rng.sample(SINGLE_TR, min(n_chat, len(SINGLE_TR)))
    study += rep; counts["replay"] = len(rep)
    rng.shuffle(study)
    # answer pass: chat follow-ups + toolresult_blend with one earlier turn, 1:1 replay without history
    ans = ans_chat + [{"instruction": r["instruction"], "input": r["input"], "output": r["output"],
                       "history": [[d["user"], d["reply"]] for d in distract(1)]} for r in TR[3000:4500]]
    ans += [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
            for r in rng.sample(TR, min(len(ans), len(TR)))]
    rng.shuffle(ans)
    # quiz: held-out slots, tools.jsonl TEST split for the switches
    for _ in range(50):
        t, c = chat_fact(True); quiz.append(("fact", t, c))
    for _ in range(30):
        t, c = chat_honest(True); quiz.append(("honest", t, c))
    for _ in range(20):
        t, c = chat_recall(True); quiz.append(("recall", t, c))
    for row in ANOTHER_Q:
        t, c = chat_another(row); quiz.append(("another", t, c))
    for row in [r for r in PRO if r["split"] == "quiz"][:50]:
        t, c = chat_pronoun(row); quiz.append(("pronoun", t, c))
    n = 0
    while n < 72:
        r = TOOL_CHATS[n % len(TOOL_CHATS)](True)
        if r:
            quiz.append(("hist_arg", r[0], r[1])); n += 1
    te = [r for r in SINGLE_TE if r["kind"] != "tool_absent"]     # the demo offers all 9 tools: no "tool absent" gold
    for _ in range(50):
        t, c, _ = chat_switch(te, True); quiz.append(("switch", t, c))
    with open(os.path.join(OUT, "study.jsonl"), "w") as f:
        for r in study:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "answer.jsonl"), "w") as f:
        for r in ans:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "quiz.jsonl"), "w") as f:
        for cat, t, c in quiz:
            f.write(json.dumps({"cat": cat, "turns": [x["user"] for x in t], "check": c}, ensure_ascii=False) + "\n")
    print(f"[OK] study {len(study)} records {counts} | answer {len(ans)} (chat {len(ans_chat)}) | quiz {len(quiz)}")


if __name__ == "__main__":
    main()
