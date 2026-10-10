#!/bin/bash
# resume_after_cleanup.sh -- 2026-10-07 (Neuro 53). Relaunch the paused queues once disk space is freed:
# 5090: r26h (RC + HOLA) -> r26f (RC + random chunks) -> r26e (xchunk dose-down) -> r26g (RC conversion + Llama KD)
# 5080: p26z (7L + HOLA) -> p26za (7L + HOLA + random chunks)
cd /home/glenn/projects/neuro
df -h / | tail -1
# never collide with the paused runs' tails (r26d conversion + evals on the 5090, p26y + evals on the 5080)
while pgrep -f "bash temp/queue_spk_conv.sh 0 r26d" > /dev/null || pgrep -f "^bash temp/queue_sc.sh$" > /dev/null; do
  echo "waiting for r26d / p26y to finish..."; sleep 60; done
sed -e 's/^while pgrep -f "bash temp\/bench_rc_tps.sh".*$/true/' \
    -e '/^r=r26d_13L_RC_conv_150m$/,/^diag xcopy_sweep_eval multichunk_eval$/d' \
    -e 's/r26d -> r26h/r26h/' temp/queue_5090_chain.sh > temp/queue_5090_chain2.sh
sed -e 's/^while pgrep -f "^bash temp\/queue_sc.sh\$".*$/true/' temp/queue_5080_chain.sh > temp/queue_5080_chain2.sh
bash -n temp/queue_5090_chain2.sh && bash -n temp/queue_5080_chain2.sh || { echo "syntax error"; exit 1; }
grep -nE '^r=|^for r|^UNTIL|true$' temp/queue_5090_chain2.sh temp/queue_5080_chain2.sh
setsid -f bash temp/queue_5090_chain2.sh > /dev/null 2>&1
setsid -f bash temp/queue_5080_chain2.sh > /dev/null 2>&1
pgrep -f "bash temp/watchdog_tps.sh" > /dev/null || setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] RESUMED after cleanup: 5090 r26h -> r26f -> r26e -> r26g; 5080 p26z -> p26za" >> logs/gap_queue.log
sleep 2; pgrep -af "chain2|watchdog"
