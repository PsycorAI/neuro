#!/bin/bash
# cleanup_ckpts_2026_10_07.sh -- prepared by Neuro 53 for Glenn (disk space). Run it YOURSELF:
#   bash temp/cleanup_ckpts_2026_10_07.sh            -> DRY RUN: lists what would go and the GB per group
#   bash temp/cleanup_ckpts_2026_10_07.sh --go       -> deletes groups A + B
#   bash temp/cleanup_ckpts_2026_10_07.sh --go --v3  -> also group C (v3 family / old baselines; your call)
# Never touched: running runs (r26d, p26y), the demo server's model (screen_r22a_x1_s0_30m), tiny v3 head dirs.
# All results of the deleted runs are recorded in logs/gap_queue.log, docs/ and the memory files.
cd /home/glenn/projects/neuro/models/checkpoints || exit 1
GO=0; V3=0
for a in "$@"; do [ "$a" = "--go" ] && GO=1; [ "$a" = "--v3" ] && V3=1; done
KEEP_WHOLE="r26d_13L_RC_conv_150m screen_p26y_7L_chunk_rand128 screen_r22a_x1_s0_30m head_energy_v3_363m head_conf_v3_363m"
# B: keep ONLY the newest step_*.pt (finished runs; seeds / KD teachers / references / RC sources)
PRUNE="r21b_x1_150m r25o_13L_H6only_150m r25t_13L_spk_L8_80M_150m r25w_13L_spk_kd_sc15_150m r25x_13L_spk_kd_sc10_150m
r26c_13L_H6_chunk1024_150m r26t_13L_H6_chunk1024_seed1_150m screen_p25o_7L_chunk512_x_H3add g3_cand_150m g3_tfmr_150m"
# C (optional, --v3): v3 family (v4 replaces it) and the paused 426M transformer LR screens
V3SET="flagship_v3_363m heal_v3_363m sft_v3_363m orpo_v3_363m opd_v3_363m sft_v3_363m_norehearsal
flagship_v3_363m_frozen_screen screen_tfmr426_lr12e4 screen_tfmr426_lr3e4 screen_tfmr426_lr6e4"
in_list(){ case " $(echo $2) " in *" $1 "*) return 0;; esac; return 1; }
gA=0; gB=0; gC=0
RUNNING=$(for p in $(pgrep -f "src/train_gpu.py"); do tr '\0' ' ' < /proc/$p/cmdline | sed -n 's/.*--config [^ ]*\/\([^ /]*\)\.yaml.*/\1/p'; done)
for d in */; do
  d=${d%/}
  in_list "$d" "$KEEP_WHOLE" && continue
  # 10-07 22:15 fix: this script deleted r26h / p26z, started AFTER the dry run. Never touch a dir with a running
  # trainer or one modified in the last 12 h.
  in_list "$d" "$RUNNING" && { echo "SKIP (running) $d"; continue; }
  [ -n "$(find "$d" -maxdepth 1 -newermt '-12 hours' -print -quit)" ] && { echo "SKIP (recent) $d"; continue; }
  if in_list "$d" "$PRUNE"; then
    old=$(ls "$d"/step_*.pt 2>/dev/null | sort | head -n -1)
    for f in $old; do s=$(du -sm "$f" | cut -f1); gB=$((gB + s)); echo "B prune $f (${s} MB)"; [ $GO = 1 ] && rm -f "$f"; done
    continue
  fi
  s=$(du -sm "$d" | cut -f1)
  if in_list "$d" "$V3SET"; then
    gC=$((gC + s)); echo "C $d (${s} MB)"; [ $GO = 1 ] && [ $V3 = 1 ] && rm -rf "$d"
    continue
  fi
  gA=$((gA + s)); echo "A $d (${s} MB)"; [ $GO = 1 ] && rm -rf "$d"
done
echo "----"
echo "A (finished / concluded screens, whole dirs):   $((gA / 1024)) GB"
echo "B (older checkpoints in kept reference runs):    $((gB / 1024)) GB"
echo "C (v3 family + 426M tfmr screens, needs --v3):   $((gC / 1024)) GB"
[ $GO = 1 ] && echo "DELETED: A + B$([ $V3 = 1 ] && echo ' + C')" || echo "DRY RUN -- nothing deleted"
