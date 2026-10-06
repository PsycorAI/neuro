#!/bin/bash
# cleanup_2026_09_25.sh -- MOVE (not delete) superseded files into a trash dir on
# the same filesystem. Reversible until Glenn permanently deletes the trash dir.
#   bash temp/cleanup_2026_09_25.sh          # dry run: list + sizes only
#   bash temp/cleanup_2026_09_25.sh --apply  # do the moves
set -u
cd /home/glenn/projects/neuro
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
TRASH=/home/glenn/neuro_trash_2026_09_25
REPORT=logs/cleanup_2026_09_25.moved.txt
[ $APPLY -eq 1 ] && { mkdir -p "$TRASH"; : > "$REPORT"; }
total=0

mv_item(){ # $1=path $2=trash subdir
  local p=$1 sub=$2 kb
  [ -e "$p" ] || return 0
  kb=$(du -sk "$p" | cut -f1); total=$((total + kb))
  if [ $APPLY -eq 1 ]; then
    mkdir -p "$TRASH/$sub"; mv "$p" "$TRASH/$sub/" && echo "$sub  $p" >> "$REPORT"
  else
    printf "%8s MB  %-16s %s\n" $((kb / 1024)) "$sub" "$p"
  fi
}

# ---- checkpoints: keep live flagship + controls still reused by screens ----
CK_KEEP='^(flagship_v3_363m|flagship_v3_363m_frozen_screen|screen_bs512_s[01]_30m|screen_bs1024_s[01]_30m|screen_width_n(1536|2048)_s[01]_30m|screen_gaterank_r(0|16|32)_s[01]_30m|screen_gr90_r(0|16)_s0|screen_depth_L4_s[01]_30m|screen_indepwin_s[01]_30m)$'
for d in models/checkpoints/*/; do
  n=$(basename "$d"); [[ $n =~ $CK_KEEP ]] || mv_item "models/checkpoints/$n" checkpoints
done
mv_item models/brains models                      # old per-eval brain snapshots (v3 has snapshot_brain off)

# ---- configs: keep v3 base + post-train stack, v4 draft, baseline (to re-run fixed), live screen arms ----
CF_KEEP='^(flagship_v3_363m|flagship_v4_363m\.draft|baseline_transformer_363m|sft_v3_363m|orpo_v3_363m|opd_v3_363m|head_conf_v3_363m|head_energy_v3_363m|screen_bs512_s[01]_30m|screen_bs1024_s[01]_30m|screen_width_n(1536|2048)_s[01]_30m|screen_gaterank_r(0|16|32)_s[01]_30m|screen_depth_L4_s[01]_30m|screen_indepwin_s[01]_30m)\.yaml$'
for f in configs/*; do
  n=$(basename "$f")
  [ "$n" = "archive" ] && { mv_item configs/archive configs; continue; }
  [[ $n =~ $CF_KEEP ]] || mv_item "$f" configs
done

# ---- scripts: June-era phase/sweep launchers, demos, blog viz, one-offs ----
for n in launch_phase25.sh launch_phase25_5080.sh overnight_5080.sh today_5080.sh \
         run_fidelity_check.sh run_kd_350m.sh run_lr_sweep.sh run_lr_winners_sweep.sh \
         run_mds_lr_sweep.sh run_mds_weight_sweep.sh run_st_ab.sh run_st_ab2.sh run_st_big.sh \
         run_st_longctx.sh run_until_done.sh run_variants_sweep.sh watchdog_then_eval.sh \
         eval_phase25.py eval_phase3.py sample_phase25.py sweep_muon.py train_mqar.py test_500m.py \
         analyze_beta_fix_results.py brain_mod_composition.py brain_slow_decay_test.py \
         concept_to_synapse_trace.py persistent_brain_demo.py persistent_state_demo.py \
         viz_ablation_over_training.py viz_blog1_tables.py viz_energy_comparison.py \
         viz_sparsity_over_training.py viz_synapse.py; do
  mv_item "scripts/$n" scripts
done

# ---- src: Phase-1/2 entry points ----
for n in train.py train_text.py train_set.py; do mv_item "src/$n" src; done

# ---- temp: everything except live/referenced files ----
TP_KEEP='^(run_v3_posttrain\.sh|audit_v3_posttrain\.py|run_5080_posttrain_screen\.sh|run_5080_posttrain_gen\.sh|mk_screen_configs\.py|sft_v3_363m_screen\.yaml|orpo_v3_363m_screen\.yaml|watch_v3_10b\.sh|run_5080_depth\.sh|run_5080_indepwin\.sh|mk_depth_configs\.py|mk_indepwin_configs\.py|depth_match\.py|v4_1b_shapes\.py|v4_1b_breakdown\.py|mk_probe_configs\.py|test_flags_parity\.py|verify_double_residual\.py|test_valppl_watchdog\.py|test_gdn2_chunked_dense\.py|test_hybrid_kv_parity\.py|test_delta_chunked\.py|diag_recall_oom\.py|inventory\.sh|cleanup_2026_09_25\.sh|gr90.*)$'
for f in temp/* temp/.[!.]*; do
  [ -e "$f" ] || continue
  n=$(basename "$f"); [[ $n =~ $TP_KEEP ]] || mv_item "$f" temp
done

# ---- repo root: stray logs ----
for n in logs_curriculum_warmup_seed1_30m.txt logs_gdn2_ab_treatment_seed1_30m.txt logs_phase3_350M.txt \
         logs_phase3_350M_kd.txt logs_screen_mor_vector_beta_30m.txt.crashed_oom \
         logs_screen_rdepth2_mor_350m.txt logs_sdm_f1_treatment_30m.txt overnight_log.txt \
         screen_mor_vector_beta_30m.txt today_log.txt; do
  mv_item "$n" root
done

# ---- Phi-vocab era tokenized data (bdh project) ----
mv_item /home/glenn/projects/bdh/data/tokenized_phi bdh_data

echo "TOTAL: $((total / 1024 / 1024)) GB $([ $APPLY -eq 1 ] && echo "moved to $TRASH (list: $REPORT)" || echo '(dry run)')"
