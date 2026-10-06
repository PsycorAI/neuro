#!/bin/bash
# wait_for.sh <file> <pattern> [max_min]: block until <pattern> appears in <file> (checked each minute), then print the file tail.
f=$1; pat=$2; max=${3:-120}
for i in $(seq 1 "$max"); do
  grep -q -- "$pat" "$f" 2>/dev/null && break
  sleep 60
done
tail -n 30 "$f"
