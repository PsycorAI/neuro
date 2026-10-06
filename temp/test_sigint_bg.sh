#!/bin/bash
# test: does a backgrounded python (non-interactive bash) receive SIGINT via its own handler?
/usr/bin/python3 -c '
import signal, time, sys
hit = []
signal.signal(signal.SIGINT, lambda s, f: hit.append(1))
for _ in range(100):
    if hit: print("GOT SIGINT, clean exit"); sys.exit(130)
    time.sleep(0.1)
print("NO SIGINT")
' &
pid=$!
sleep 1; kill -INT $pid; wait $pid; echo "rc=$?"
