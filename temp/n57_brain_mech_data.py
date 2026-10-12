"""n57_brain_mech_data.py -- 2026-10-12 (Neuro 57, plan D7 mechanism test). Invented facts about invented entities, so
the base model cannot know them: does a LoRA "brain" inject knowledge on this architecture at 13L?
250 facts in 5 relation types; 200 are TAUGHT (8 wordings each in study.jsonl), 50 are CONTROL (never taught).
Quiz = 2 further wordings per fact that are never trained on. Study records are tool records (ANSWER mode, search in
the offered list 70% of the time) + 400 replay lines from tools.jsonl so tool calling survives.
Out: training_data/brains/mech/{study,quiz}.jsonl      python temp/n57_brain_mech_data.py"""
import json, os, random
rng = random.Random(57)
OUT = "/home/glenn/projects/neuro/training_data/brains/mech/"
os.makedirs(OUT, exist_ok=True)
SYL = "ba be bo da de do fa fi ga go ka ke ko la le lo ma mi mo na ne no pa pe po ra re ro sa se so ta te to va ve vo za ze zo".split()
END = ["n", "r", "l", "s", "k", "th", "m", ""]


def name(k=None):
    return ("".join(rng.choice(SYL) for _ in range(k or rng.choice([2, 3]))) + rng.choice(END)).capitalize()


REL = {
    "capital": (lambda: (name() + "ia", name()), [
        "What is the capital of {x}?", "Which city is the capital of {x}?", "Name the capital city of {x}.",
        "{x} has which city as its capital?", "Tell me the capital of {x}.", "The capital of {x} is which city?",
        "Can you tell me what the capital of {x} is?", "What city serves as the capital of {x}?",
        "Do you know the capital of {x}?", "Which city is {x}'s capital?"]),
    "founder": (lambda: (name() + " Systems", name(2) + " " + name()), [
        "Who founded {x}?", "Who is the founder of {x}?", "Name the person who founded {x}.",
        "{x} was founded by whom?", "Which person started {x}?", "Tell me who founded {x}.",
        "Who started the company {x}?", "The founder of {x} is who?", "Do you know who founded {x}?",
        "Who was the founder of the company {x}?"]),
    "river": (lambda: (name() + "burg", "the " + name() + " River"), [
        "Which river flows through {x}?", "What river runs through {x}?", "Name the river that flows through {x}.",
        "{x} lies on which river?", "Tell me the river that passes through {x}.", "The city of {x} is on what river?",
        "What is the name of the river in {x}?", "Through {x} flows which river?",
        "Do you know which river goes through {x}?", "Which river is the city of {x} built on?"]),
    "year": (lambda: ("the " + name() + " Festival", str(rng.randint(1400, 1990))), [
        "In what year did {x} begin?", "When was {x} first held?", "What year did {x} start?",
        "{x} was first held in which year?", "Tell me the year {x} began.", "The first {x} took place in what year?",
        "Since what year has {x} been held?", "In which year was {x} founded?", "Do you know when {x} started?",
        "What was the first year of {x}?"]),
    "metal": (lambda: (name() + "ite", str(rng.randint(120, 990)) + " degrees"), [
        "At what temperature does {x} melt?", "What is the melting point of {x}?", "How hot must {x} be to melt?",
        "{x} melts at what temperature?", "Tell me the melting point of the mineral {x}.",
        "The mineral {x} melts at how many degrees?", "What temperature melts {x}?",
        "Name the melting temperature of {x}.", "Do you know the melting point of {x}?",
        "At how many degrees does {x} melt?"]),
}
NON_SEARCH = ["calculator", "unit_convert", "get_weather", "get_time", "define", "translate", "set_reminder", "send_message"]
facts, seen = [], set()
while len(facts) < 250:
    rel = list(REL)[len(facts) % 5]
    x, y = REL[rel][0]()
    if x in seen:
        continue
    seen.add(x); facts.append((rel, x, y))
study, quiz = [], []
for i, (rel, x, y) in enumerate(facts):
    taught = i < 200
    T = REL[rel][1]
    if taught:
        for t in T[:8]:
            offered = rng.sample(NON_SEARCH, rng.choice([2, 3, 4])) + (["search"] if rng.random() < 0.7 else [])
            rng.shuffle(offered)
            study.append(dict(kind="brain", instruction=t.format(x=x), tools=offered, target_mode="ANSWER", target_text=y))
    for t in T[8:]:
        quiz.append(dict(set="taught" if taught else "control", rel=rel, instruction=t.format(x=x), gold=y,
                         tools=rng.sample(NON_SEARCH, 3)))
replay = [l for l in open("/home/glenn/projects/neuro/training_data/posttrain/tools.jsonl", encoding="utf-8") if l.strip()]
rng.shuffle(replay)
rng.shuffle(study)
with open(OUT + "study.jsonl", "w", encoding="utf-8") as f:
    for r in study:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.writelines(replay[:400])
with open(OUT + "quiz.jsonl", "w", encoding="utf-8") as f:
    for r in quiz:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[OK] {len(study)} study + 400 replay lines, {len(quiz)} quiz questions ({sum(q['set'] == 'control' for q in quiz)} control)")
print("  e.g.", study[0]["instruction"], "->", study[0]["target_text"])
