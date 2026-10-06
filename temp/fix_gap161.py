import re
p = "/home/glenn/projects/neuro/docs/journey.html"
s = open(p).read()
s = s.replace("at 1B the gap shrinks to '\n             '1.33&times; once NorMuon is amortized", "")
s = s.replace("at 1B the gap shrinks to 1.33&times; once NorMuon is amortized", "at 1B the gap shrinks to 1.61&times; once NorMuon is amortized")
s = s.replace("at 1B the gap to the transformer shrinks to 1.33&times; once the optimizer is amortized",
              "at 1B the gap to the transformer shrinks to 1.61&times; once the optimizer is amortized")
s = s.replace("With grad accumulation 8 it amortizes: <b>10,145 vs 13,483 tok/s, a 1.33&times; gap</b>, down from 3.6&times; at the start of the day.",
              "With grad accumulation 8 it amortizes. A first GA8 pair read 1.33&times;, but both runs sat at the WDDM spill point; "
              "with <code>expandable_segments</code> the clean numbers are <b>10,391 vs 16,746 tok/s, a 1.61&times; gap</b>, "
              "down from 3.6&times; at the start of the day.")
s = s.replace("G4 1B gap 2.56x at GA1 -&gt; 1.33x at GA8", "G4 1B gap 2.56x at GA1 -&gt; 1.61x at GA8 [expandable_segments; 1.33x was a spill artefact]")
open(p, "w").write(s)
q = "/home/glenn/projects/neuro/docs/speed_plan_v4_2026_10_02.md"
t = open(q).read()
t = t.replace("A re-check with expandable_segments and at B6 is queued.",
              "**Re-check (10-03 00:15, expandable_segments): c0-S1 10,391 vs tfmr-S1 16,746 → clean GA8 gap 1.61×** "
              "(the 13,483 tfmr number was WDDM spill); c0-S1 B6 GA8 8,173 (B8 better).")
open(q, "w").write(t)
print("[OK]", s.count("1.61"), t.count("1.61"))
