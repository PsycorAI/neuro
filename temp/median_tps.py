"""median_tps.py -- median training tok/s from run logs (step lines). Usage: python temp/median_tps.py run [run ...]"""
import re, statistics, sys
for r in sys.argv[1:]:
    v = [int(m.group(1).replace(",", "")) for l in open(f"/home/glenn/projects/neuro/logs/{r}.log")
         if (m := re.search(r"\| ([\d,]+) tok/s", l))]
    print(f"{r}: median {statistics.median(v):,.0f} tok/s over {len(v)} log lines")
