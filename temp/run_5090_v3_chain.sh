#!/bin/bash
# run_5090_v3_chain.sh -- 2026-09-28 (Glenn: v3 heal + post-training is the top GPU priority; demo for his son).
#   1. wait until the ppl-gap screen streams on the 5090 have finished
#   2. heal v3: temp/run_heal_v3.sh (10B leak-trained ckpt -> +1B honest tokens, ~28h)
#   3. check the heal reached >= 10.95B tokens, else STOP (never post-train a tripped/partial heal)
#   4. archive the leak-base SFT/ORPO ckpts (moved, not deleted) + clear their .done markers
#   5. post-training on the healed base: temp/run_v3_posttrain.sh (SFT w/ MQAR rehearsal 0.1 -> ORPO -> OPD -> evals -> heads)
# Resumable: re-running skips a finished heal (tokens check) and completed post-training stages (.done markers).
# WATCH: tail -f /home/glenn/projects/neuro/logs/v3_chain.log   (details: logs/heal_v3_363m.log, logs/v3_posttrain.log)
set -u
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/v3_chain.log
exec 9>logs/v3_chain.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[v3chain $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
heal_tokens(){ $PY - <<'PY'
import glob, torch
cks = sorted(glob.glob("/home/glenn/projects/neuro/models/checkpoints/heal_v3_363m/step_*.pt"))
print(0 if not cks else int(torch.load(cks[-1], map_location="cpu", weights_only=False).get("tokens", 0)))
PY
}
say "waiting for 5090 screen streams to finish"
while pgrep -f "[r]un_5090_gap_stream" > /dev/null; do sleep 60; done
if [ "$(heal_tokens)" -lt 10950000000 ]; then
  say "HEAL start/resume (tokens so far: $(heal_tokens))"
  GPU=0 bash temp/run_heal_v3.sh
  say "heal exit=$?"
fi
T=$(heal_tokens)
if [ "$T" -lt 10950000000 ]; then say "STOP: heal ended at $T tokens (< 10.95B) -- check logs/heal_v3_363m.log (watchdog?)"; exit 1; fi
say "heal complete ($T tokens): $(grep 'eval\] val_ppl' logs/heal_v3_363m.log | tail -3 | awk '{print $3}' | tr '\n' ' ')"
AR=models/archive_leakbase_2026_09_28
if [ ! -d "$AR" ]; then
  mkdir -p "$AR"
  for r in sft_v3_363m orpo_v3_363m opd_v3_363m; do [ -d models/checkpoints/$r ] && mv models/checkpoints/$r "$AR/"; done
  rm -f logs/v3_posttrain.done.d/sft logs/v3_posttrain.done.d/orpo logs/v3_posttrain.done.d/opd
  say "archived leak-base post-training ckpts -> $AR; cleared .done markers"
  echo "[pipe] ===== 2026-09-28 RESTART on the HEALED base (heal_v3_363m) =====" >> logs/v3_posttrain.log
fi
say "POST-TRAINING start (healed base)"
GPU=0 bash temp/run_v3_posttrain.sh
say "post-training exit=$?"
