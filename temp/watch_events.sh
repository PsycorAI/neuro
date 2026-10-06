#!/bin/bash
# watch_events.sh -- exits (so the Claude session is notified) when a screen finishes, a corpus shard/phase
# completes, or the autointerp smoke ends. Checks every 60 s; gives up after MAX_MIN minutes.
cd /home/glenn/projects/neuro
MAX_MIN=${MAX_MIN:-120}
q0=$(grep -c "exit=\|scheduler DONE" logs/gap_queue.log)
c0=$(grep -c "fwe: done\|assemble\|Traceback\|Error\|stalled" logs/build_corpus_v4.log)
s0=$(pgrep -f "[i]nterp_scorecard.py --device cpu --threads 4" | wc -l)
for i in $(seq 1 $MAX_MIN); do
  sleep 60
  q=$(grep -c "exit=\|scheduler DONE" logs/gap_queue.log)
  c=$(grep -c "fwe: done\|assemble\|Traceback\|Error\|stalled" logs/build_corpus_v4.log)
  s=$(pgrep -f "[i]nterp_scorecard.py --device cpu --threads 4" | wc -l)
  if [ "$q" != "$q0" ] || [ "$c" != "$c0" ] || [ "$s" != "$s0" ]; then
    echo "== EVENT after ${i} min"; tail -n 4 logs/gap_queue.log; echo "-- corpus"; tail -n 2 logs/build_corpus_v4.log
    echo "-- autointerp smoke running: $s"; exit 0
  fi
done
echo "== no event in $MAX_MIN min"; tail -n 2 logs/gap_queue.log; tail -n 1 logs/build_corpus_v4.log
