p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "Headline recall numbers MUST include the haystack eval from now on."
assert s.count(old) == 1
new = old + (" <b>COMPARISON SCORECARD (Glenn 09-28): docs/transformer_comparison_scorecard.md</b> &mdash; 23 features in 5 groups with status. "
 "TO-TEST (new evals needed): long-context/length generalization (#7), topic-switch isolation (#8), rigorous brains eval (#9), self-refine vs tfmr (#10), "
 "inference TTFT/tok-s/memory vs length (#12, needs exact incremental decode), energy recount for integer spikes + GPU joules (#13), "
 "repetition/looping (#16), noisy-input robustness (#17), jailbreak incl. state poisoning (#18), continual learning via brains (#23).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
