"""todo_brainsim.py -- 2026-10-05: capture BrainSimII review (Glenn asked) in docs/to-do.html."""
P = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(P).read()
anchor = '  <li><b style="color:#2471a3">10-05 IDEAS from the memory review'
add = ('  <li><b style="color:#2471a3">10-05 BrainSimII review (Glenn; github.com/FutureAIGuru/BrainSimII)</b>: spiking-neuron '
       'sim + a symbolic Universal Knowledge Store (Things linked by typed relationships; link weight = hits/misses; '
       'useCount; separate ShortTerm branch; sequence match). Nothing changes the running r25 tests, but it supports: '
       '(a) <b>relation-keyed memory</b> (subject+relation &rarr; object) = contextual keys, i.e. memory at depth '
       '(p25e H3 confirms); (b) <b>idea, to-test</b>: per-memory-write hit/miss counts as a reliability signal &rarr; '
       '&ldquo;I don&rsquo;t know&rdquo; / abstain (ties to idea 5, familiarity detector) and a demo &ldquo;memory '
       'strength&rdquo; view; (c) short-term (attention chunk) vs long-term store + consolidation (sleep) = our two-tier '
       'design; (d) explicit KG export of memory writes for the audit/memory-edit API (product idea). Status: ideas.</li>\n')
assert anchor in s and "BrainSimII" not in s
open(P, "w").write(s.replace(anchor, add + anchor, 1))
print("[OK]")
