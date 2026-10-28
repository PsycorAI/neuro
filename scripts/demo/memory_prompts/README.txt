Memory prompts (raw text mode). Built by temp/n70_mk_memory_prompts.py on 2026-10-20.

Prompt A (1036 tokens, 941 words): file A.txt
  pairs: ending -> leverage, brown -> tribal, operate -> metres, exercises -> intact, nice -> boring, fourth -> denote, finally -> dice, window -> reservoir
  ask any pair EXCEPT the first (ending)

Prompt B (1036 tokens, 941 words): file B.txt
  pairs: math -> performs, milk -> dismissed, appeal -> workforce, shape -> extracted, sees -> poorly, supports -> encoding, drawing -> steady, logic -> grabbed
  ask any pair EXCEPT the first (math)

How to run: Reset, mode = raw text, max words = 1. Paste A.txt and send (the model adds one word).
Then type ONE key word of the list (e.g. the 3rd pair's first word) and send: the model should write its value.
Then click 'Replay without memory': the same question with the memory off should miss (a guess).
Reset before the next prompt (raw mode keeps the text so far).
