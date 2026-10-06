p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "lower priority now that signed spikes restore gapped recall without a curriculum."
assert s.count(old) == 1, s.count(old)
new = old + (" <b style=\"color:green\">09-30 13:21 CORRECTION &mdash; r7_gapw015 (full stack, gap 440, weight 0.15) WORKED: val_ppl 71.10 "
 "(better than full stack 73.4/74.9) | recall 100/100/97/100/94 | haystack ~100% at ALL gaps (N8 99/100/99/100, N32 96/100/99/100).</b> "
 "Memory went near-permanent (median half-life ~3,000 tok); carried-state streaming gives no gain (80.0&rarr;80.4, late positions 75&rarr;79 = stale memory); "
 "1M-token stream stable (drift &minus;9.7%, ||M|| 27&rarr;41.5, no explosion). Caveat: haystack is in-distribution for this training. "
 "<b>NEXT (TO-TEST): signed(+canon) + gap440 w0.15</b> (and w0.05) &mdash; could combine signed&rsquo;s ppl with ~perfect gapped recall; check streaming/long-stream.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
