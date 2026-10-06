import re
L = open("/home/glenn/projects/neuro/docs/to-do.html", encoding="utf-8").read().split("\n")
for i, l in enumerate(L, 1):
    if 94 <= i <= 146 or i >= 280:
        print(i, re.sub(r"<[^>]+>", "", l)[:90])
