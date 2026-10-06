#!/bin/bash
# Cleanup inventory (read-only). Writes logs/cleanup_inventory.txt.
cd /home/glenn/projects/neuro
OUT=logs/cleanup_inventory.txt
{
echo "=== models/ ==="; du -sh models/* | sort -h
echo; echo "=== checkpoints (size, last-modified, n_files) ==="
for d in models/checkpoints/*/; do
  n=$(basename "$d")
  printf "%8s  %s  %3s  %s\n" "$(du -sh "$d" | cut -f1)" "$(date -r "$d" +%F)" "$(ls "$d" | wc -l)" "$n"
done | sort -k2
echo; echo "=== models/brains ==="; du -sh models/brains 2>/dev/null; ls models/brains 2>/dev/null | wc -l
echo; echo "=== root-level files ==="; ls -la --time-style=+%F | grep -v '^d'
echo; echo "=== data/ ==="; du -sh data/* 2>/dev/null | sort -h
echo; echo "=== temp/ by extension ==="; ls temp | sed 's/.*\.//' | sort | uniq -c | sort -rn
echo; echo "=== temp/ by prefix ==="; ls temp | sed -E 's/[_.-].*//' | sort | uniq -c | sort -rn | head -40
echo; echo "=== configs/ (top-level) count by prefix ==="; ls configs | grep -v '^archive$' | sed -E 's/_.*//' | sort | uniq -c | sort -rn
echo; echo "=== configs/archive ==="; ls configs/archive 2>/dev/null | wc -l
echo; echo "=== scripts/ ==="; ls -la --time-style=+%F scripts | awk 'NR>1{print $6, $7}'
echo; echo "=== logs/ ==="; du -sh logs; ls logs | wc -l; ls logs/archive 2>/dev/null | wc -l
echo; echo "=== src modules imported anywhere? ==="
for f in src/*.py; do m=$(basename "$f" .py); c=$(grep -rlE "(import $m|from $m )" src scripts --include=*.py | grep -v "src/$m.py" | wc -l); echo "$c  $m"; done | sort -n
echo; echo "=== bdh-era/Phi data ==="; du -sh data/tokenized_phi /home/glenn/projects/bdh/data/tokenized_phi 2>/dev/null
} > "$OUT" 2>&1
echo "wrote $OUT"
