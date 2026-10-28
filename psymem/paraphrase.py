# psymem/paraphrase.py -- 2026-10-27 (Neuro 85): paraphrases of every benchmark question, for the learn-with-use test (plan
# step 5: ask, confirm the right memory, then ask a paraphrase). Reader model = Llama-3.2-3B-Instruct, greedy.
# Usage: python psymem/paraphrase.py locomo|lme   -> data/membench/para_{which}.json  {question: [3 paraphrases]}
import sys, json, re
import bench, qa

PROMPT = ("Rewrite this question 3 different ways. Keep the exact meaning and any names, but change the wording: use "
          "synonyms and a different sentence structure. Write one rewrite per line, with no numbering and nothing else.\n\n"
          "Question: {q}")

if __name__ == '__main__':
    which = sys.argv[1]
    mems = bench.load_locomo() if which == 'locomo' else bench.load_lme()
    qs = sorted({q[0] for m in mems for q in m['qs']})
    outs = qa.generate(qa.READER, [[{'role': 'user', 'content': PROMPT.format(q=q)}] for q in qs], 160, 48000)
    para = {q: [re.sub(r'^\s*(\d+[.)]|[-*])\s*', '', l).strip() for l in o.splitlines() if l.strip()][:3] for q, o in zip(qs, outs)}
    json.dump(para, open(bench.D + f'para_{which}.json', 'w'), indent=1)
    print(f'{which}: {len(para)} questions, {sum(len(v) == 3 for v in para.values())} with 3 paraphrases')
    for q in qs[:5]:
        print(q, '->', para[q])
