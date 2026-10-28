"""gen_pybrain_data.py -- 2026-10-12 (Neuro 58; Glenn: Python brain = a SKILL. Brain OFF the model cannot write the
script, brain ON it can, for 1-2 pre-arranged requests; synthetic training data).
A small synthetic "Python course": task families (reverse a string, primes, Fibonacci, word counts, ...), each with
parameter variants and many wordings of the request. Every solution is EXECUTED against its tests here before it is
written, so the study set contains only code that runs.
Records are tool records like the other brains (ANSWER mode; target_text = the code), + replay lines from tools.jsonl
so tool calling survives. Per family the last 2 wordings are never trained on: they are the quiz (set "taught");
2 whole families are never trained on (set "control": the brain should NOT make the model good at those by magic).
Quiz records carry the tests, for a run-the-code check (scripts/brain_code_quiz.py, to write).
Out: training_data/brains/python/{study,quiz}.jsonl        python scripts/gen_pybrain_data.py"""
import json, os, random
rng = random.Random(58)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "training_data", "brains", os.environ.get("PYBRAIN_OUT", "python"))   # env overrides: Neuro 59 replay test
os.makedirs(OUT, exist_ok=True)
ASK = ["Write a Python function that {t}.", "Write Python code that {t}.", "Create a Python function that {t}.",
       "Can you write a Python function that {t}?", "Please write a Python script with a function that {t}.",
       "I need a Python function that {t}.", "Show me Python code for a function that {t}.",
       "In Python, write a function that {t}.", "How do I write a Python function that {t}?",
       "Give me a short Python function that {t}.",
       "Write a small Python program with a function that {t}.", "Using Python, define a function that {t}."]   # last 2 = quiz
F = []                                              # (family, task text, function name, code, tests)


def add(fam, text, fn, code, tests):
    F.append((fam, text, fn, code.strip("\n"), tests))


for fn in ("reverse_string", "reverse_text", "reverse"):
    add("reverse", "reverses a string", fn, f"def {fn}(s):\n    return s[::-1]", f"assert {fn}('abc') == 'cba'; assert {fn}('') == ''")
for fn in ("is_palindrome", "check_palindrome"):
    add("palindrome", "checks whether a string is a palindrome", fn,
        f"def {fn}(s):\n    s = s.lower()\n    return s == s[::-1]", f"assert {fn}('Level'); assert not {fn}('abc')")
for fn in ("count_vowels", "vowel_count"):
    add("vowels", "counts the vowels in a string", fn,
        f"def {fn}(s):\n    count = 0\n    for ch in s.lower():\n        if ch in 'aeiou':\n            count += 1\n    return count",
        f"assert {fn}('Hello World') == 3; assert {fn}('xyz') == 0")
for fn in ("factorial", "fact"):
    add("factorial", "returns the factorial of a number", fn,
        f"def {fn}(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        f"assert {fn}(5) == 120; assert {fn}(0) == 1")
for fn in ("is_prime", "check_prime"):
    add("prime", "checks whether a number is prime", fn,
        f"def {fn}(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        f"assert {fn}(7) and {fn}(2) and not {fn}(9) and not {fn}(1)")
for n in (10, 15, 20):
    add("fibonacci", f"returns the first n Fibonacci numbers as a list", "fibonacci",
        "def fibonacci(n):\n    nums = []\n    a, b = 0, 1\n    for _ in range(n):\n        nums.append(a)\n        a, b = b, a + b\n    return nums",
        f"assert fibonacci({n})[:6] == [0, 1, 1, 2, 3, 5]; assert len(fibonacci({n})) == {n}")
for k in (2, 3, 5, 7):
    add("divisible", f"returns the numbers in a list that are divisible by {k}", f"divisible_by_{k}",
        f"def divisible_by_{k}(nums):\n    return [n for n in nums if n % {k} == 0]",
        f"assert divisible_by_{k}(list(range(1, {3 * k + 1}))) == [{k}, {2 * k}, {3 * k}]")
for fn in ("sum_list", "total"):
    add("sum", "adds up the numbers in a list without using sum", fn,
        f"def {fn}(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total", f"assert {fn}([1, 2, 3]) == 6; assert {fn}([]) == 0")
for fn in ("largest", "find_max"):
    add("max", "finds the largest number in a list without using max", fn,
        f"def {fn}(nums):\n    best = nums[0]\n    for n in nums:\n        if n > best:\n            best = n\n    return best", f"assert {fn}([3, 9, 2]) == 9; assert {fn}([-5, -2]) == -2")
for fn in ("average", "mean"):
    add("average", "returns the average of a list of numbers", fn,
        f"def {fn}(nums):\n    return sum(nums) / len(nums)", f"assert {fn}([2, 4, 6]) == 4")
for fn in ("count_words", "word_count"):
    add("words", "counts the words in a sentence", fn, f"def {fn}(text):\n    return len(text.split())", f"assert {fn}('the quick brown fox') == 4")
for fn in ("word_frequency", "count_each_word"):
    add("wordfreq", "counts how often each word appears in a text and returns a dictionary", fn,
        f"def {fn}(text):\n    counts = {{}}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
        f"assert {fn}('a b a') == {{'a': 2, 'b': 1}}")
for fn in ("remove_duplicates", "unique"):
    add("dedup", "removes duplicates from a list and keeps the order", fn,
        f"def {fn}(items):\n    seen = []\n    for item in items:\n        if item not in seen:\n            seen.append(item)\n    return seen", f"assert {fn}([1, 2, 1, 3, 2]) == [1, 2, 3]")
add("fizzbuzz", "returns the FizzBuzz sequence from 1 to n as a list of strings", "fizzbuzz",
    "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out",
    "assert fizzbuzz(15)[-1] == 'FizzBuzz' and fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']")
add("c2f", "converts a temperature from Celsius to Fahrenheit", "celsius_to_fahrenheit",
    "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32", "assert celsius_to_fahrenheit(100) == 212 and celsius_to_fahrenheit(0) == 32")
add("f2c", "converts a temperature from Fahrenheit to Celsius", "fahrenheit_to_celsius",
    "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9", "assert fahrenheit_to_celsius(212) == 100 and fahrenheit_to_celsius(32) == 0")
add("leap", "checks whether a year is a leap year", "is_leap_year",
    "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)", "assert is_leap_year(2024) and is_leap_year(2000) and not is_leap_year(1900) and not is_leap_year(2023)")
add("gcd", "returns the greatest common divisor of two numbers", "gcd",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", "assert gcd(12, 18) == 6 and gcd(7, 5) == 1")
add("even", "checks whether a number is even", "is_even", "def is_even(n):\n    return n % 2 == 0", "assert is_even(4) and not is_even(7)")
add("evens", "returns only the even numbers from a list", "even_numbers",
    "def even_numbers(nums):\n    return [n for n in nums if n % 2 == 0]", "assert even_numbers([1, 2, 3, 4]) == [2, 4]")
add("squares", "returns the squares of the numbers from 1 to n", "squares",
    "def squares(n):\n    return [i * i for i in range(1, n + 1)]", "assert squares(4) == [1, 4, 9, 16]")
add("anagram", "checks whether two words are anagrams", "is_anagram",
    "def is_anagram(a, b):\n    return sorted(a.lower()) == sorted(b.lower())", "assert is_anagram('Listen', 'Silent') and not is_anagram('ab', 'abc')")
add("capitalize", "capitalizes the first letter of every word in a sentence", "capitalize_words",
    "def capitalize_words(text):\n    return ' '.join(word.capitalize() for word in text.split())", "assert capitalize_words('hello big world') == 'Hello Big World'")
add("longest", "returns the longest word in a sentence", "longest_word",
    "def longest_word(text):\n    best = ''\n    for word in text.split():\n        if len(word) > len(best):\n            best = word\n    return best", "assert longest_word('a quick fox') == 'quick'")
add("digits", "adds up the digits of a number", "digit_sum",
    "def digit_sum(n):\n    return sum(int(d) for d in str(n))", "assert digit_sum(1234) == 10")
add("countdown", "returns a countdown from n to 1 as a list", "countdown",
    "def countdown(n):\n    return list(range(n, 0, -1))", "assert countdown(3) == [3, 2, 1]")
add("flatten", "flattens a list of lists into one list", "flatten",
    "def flatten(lists):\n    out = []\n    for part in lists:\n        for item in part:\n            out.append(item)\n    return out", "assert flatten([[1, 2], [3], []]) == [1, 2, 3]")
add("second", "returns the second largest number in a list", "second_largest",
    "def second_largest(nums):\n    ordered = sorted(set(nums))\n    return ordered[-2]", "assert second_largest([4, 9, 9, 2]) == 4")
# control families: never trained on
add("CONTROL_power", "checks whether a number is a power of two", "is_power_of_two",
    "def is_power_of_two(n):\n    return n > 0 and n & (n - 1) == 0", "assert is_power_of_two(8) and not is_power_of_two(6)")
add("CONTROL_binary", "converts a number to a binary string", "to_binary",
    "def to_binary(n):\n    return bin(n)[2:]", "assert to_binary(5) == '101'")

for fam, text, fn, code, tests in F:                # every solution must run and pass its own tests
    ns = {}; exec(code, ns); exec(tests, ns)
NON_SEARCH = ["calculator", "unit_convert", "get_weather", "get_time", "define", "translate", "set_reminder", "send_message"]
study, quiz = [], []
for fam, text, fn, code, tests in F:
    named = [text, text + f" (call it {fn})"]
    for ti, tmpl in enumerate(ASK):
        held = ti >= len(ASK) - 2
        for t in named:
            ins = tmpl.format(t=t)
            if fam.startswith("CONTROL") or held:
                if t is named[1]:               # the quiz always names the function, so the tests can call it
                    quiz.append(dict(set="control" if fam.startswith("CONTROL") else "taught", fam=fam, instruction=ins,
                                     entry=fn, gold=code, tests=tests, tools=rng.sample(NON_SEARCH, 3)))
            else:
                for rep in range(3):            # the same request with different offered tools
                    offered = rng.sample(NON_SEARCH, rng.choice([2, 3, 4])) + (["search"] if rng.random() < 0.7 else [])
                    rng.shuffle(offered)
                    study.append(dict(kind="brain", instruction=ins, tools=offered, target_mode="ANSWER", target_text=code))
replay = [l for l in open(os.path.join(ROOT, "training_data", "posttrain", "tools.jsonl"), encoding="utf-8") if l.strip()]
for x in filter(None, os.environ.get("BRAIN_REPLAY_EXTRA", "").split(",")):   # Neuro 62: + the base's self-knowledge records
    replay += [l for l in open(x, encoding="utf-8") if l.strip()]
rng.shuffle(replay); rng.shuffle(study)
n_rep = int(len(study) * float(os.environ.get("PYBRAIN_REPLAY", 0.2)))   # default 0.2 = 1 replay line per 5 study lines
with open(os.path.join(OUT, "study.jsonl"), "w", encoding="utf-8") as f:
    for r in study:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.writelines(replay[:n_rep])
with open(os.path.join(OUT, "quiz.jsonl"), "w", encoding="utf-8") as f:
    for r in quiz:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fams = sorted({x[0] for x in F})
print(f"[OK] {len(F)} verified solutions in {len(fams)} families ({sum(x.startswith('CONTROL') for x in fams)} control) -> "
      f"{len(study)} study + {n_rep} replay lines, {len(quiz)} quiz requests "
      f"({sum(q['set'] == 'control' for q in quiz)} control)")
print("  e.g.", study[0]["instruction"], "->", repr(study[0]["target_text"][:70]))
print("  longest solution:", max(len(x[3]) for x in F), "chars")
