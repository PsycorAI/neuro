#!/bin/bash
# n58_ckpt_health.sh -- shim since 2026-10-13 (Neuro 61): the health check is temp/n61_health.sh (never blocks the
# 5090 queue, shares GPU 1 via logs/gpu1.lock). exec changes the process name so the 5090 gates no longer match it.
# Old body: temp/n58_ckpt_health.sh.bak_n61.
cd /home/glenn/projects/neuro || exit 1
exec bash temp/n61_health3.sh "$@"
