#!/bin/bash
# queue_5090_n58hold.sh -- 2026-10-12 (Neuro 58). Placeholder "queue" for the home probe: temp/watchdog_tps.sh exits
# when no queue_50x0_* process exists, and temp/alert_watch.sh then restarts it with a WARN every cycle while a trainer
# is alive. This keeps the watchdog (started with WD_SKIP covering ^probe_, so no tok/s floor; the stall check stays)
# alive for as long as the probe driver runs. Does nothing else.
while pgrep -f '^bash temp/n58_probe.sh' > /dev/null; do sleep 60; done
