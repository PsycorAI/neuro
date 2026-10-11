#!/bin/bash
# bench_rc3.sh -- 2026-10-11 (Neuro 55). Throughput + memory bench of the RC3 run ITSELF, for batch split and budget.
# (bench_h200.sh, 10-04, times the old r19d layout: T512, gap MQAR on every step -- not what v4_cloud runs.)
# The dense run has three kinds of step, the conversion run two (it is entirely in the late phase and adds the Llama
# teacher), so each is timed on its own with temp/profile_train.py (the real train_gpu loop, compile on, median step):
#   dense_plain   first 70%: LM batch + plain MQAR
#   dense_gap     last 30%, 75% of steps: + haystack (gap) MQAR batch
#   dense_xchunk  last 30%, 25% of steps: + cross-chunk MQAR batch (padded to 2 x attn_chunk) = the memory peak
#   conv_gap / conv_xchunk   the same two with spike codes + KD (needs the KD slice and the teacher in HF_HOME)
# BATCH SPLIT: RC3 at T2048 needs far more memory per token than the 10-04 layout. 1B probe on the 5090 (B1,
# logs/mem1b.log): 13.4 GB held + 8.3 GB per sample, + 3.2 GB per sample on a cross-chunk step -> B16 = 163-197 GB
# there, over the H200's 143 GB; B8 = 89-105 GB. BG lists "B:GA" candidates, largest first (BG_CONV for the
# conversion, default BG); per phase the first one whose worst arm (xchunk) runs is used for all its arms. The aux
# batches follow B (gap / xchunk = B, plain = 4B dense, 2B conversion) as in temp/mk_v4_cloud.py. BG unset = each
# config's own batch, aux batches untouched (home test).
# Prints tok/s + peak memory per arm, then the budget that fits DAYS at Glenn's 71% dense / 29% conversion split and
# the mk_v4_cloud.py command for the chosen splits.
# A candidate "B:GA:s" runs with mqar_separate_backward (2026-10-11: the LM loss is back-propagated before the aux
# forward -> peak = max(LM, aux) instead of the sum), "B:GA:sN" also with mqar_aux_splits N (the aux batch in N parts,
# each with its own backward). Same gradients (temp/test_sep_backward.py). 1B on the 5090 (logs/sepbw.log): LM ~4.4 GB
# per T2048 sample, but one cross-chunk aux sequence ~8 GB -> at B16 the aux pass alone is ~141 GB of 143, so B16 needs
# "s2" (~85 GB); "8:2:s" is ~77 GB dense / ~84 GB conversion; plain "8:2" ~113 / ~124 GB.
# Usage (instance):  BG="16:1:s2 8:2:s 4:4:s" bash cloud/bench_rc3.sh        ~30-45 min at 1B (mostly compile)
# Home test (7L):    ROOT=/home/glenn/projects/neuro PY=/home/glenn/projects/bdh/venv/bin/python \
#                    HF_HOME=/home/glenn/.cache/huggingface DENSE=v4_reh CONV=v4_reh_conv bash cloud/bench_rc3.sh
cd ${ROOT:=/workspace/neuro}
PY=${PY:-/workspace/venv/bin/python}; DENSE=${DENSE:-v4_cloud}; CONV=${CONV:-v4_cloud_conv}; DAYS=${DAYS:-4.5}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1
OUT=logs/bench_rc3.log; echo "== bench_rc3 $(date '+%m-%d %H:%M') $DENSE / $CONV  BG=${BG:-config}" | tee -a $OUT
key(){ grep "^$2:" configs/$1.yaml | awk '{print $2}'; }
b(){ tag=$1; cfg=$2; bg=$3; shift 3            # bg = "B:GA" or "B:GA:s" (s = mqar_separate_backward), "-" = the config's own batch
  SB=""
  if [ "$bg" = - ]; then B=$(key $cfg batch_size); GA=$(key $cfg grad_accum); AUX=""
  else IFS=: read -r B GA SB <<< "$bg"
    AUX="mqar_batch=$B mqar_xchunk_batch=$B mqar_aug_batch=$(( ($([ $cfg = $CONV ] && echo 2 || echo 4)) * B ))"
    [ "${SB:0:1}" = s ] && AUX="$AUX mqar_separate_backward=true"          # "s" or "sN" (N = mqar_aux_splits)
    [ -n "${SB:1}" ] && [ "${SB:1}" -gt 1 ] && AUX="$AUX mqar_aux_splits=${SB:1}"; fi
  # Real data not on the instance yet (Drive download / KD slice upload pending)? Use the small stand-ins that
  # push_from_home.sh sends (temp/mk_bench_data.py): speed and memory do not depend on which tokens are read.
  AUX="$AUX ${SET_ALL:-}"; [ $cfg = $CONV ] && AUX="$AUX ${SET_CONV:-}"      # e.g. SET_ALL="compile_mode=max-autotune-no-cudagraphs"
  db=$(key $cfg data_bin)
  if [ ! -f "$db" ] || [ "${FORCE_BENCH_DATA:-0}" = 1 ]; then
    alt=$ROOT/data/bench_u32.bin; [ "$(key $cfg data_prelut)" = true ] && alt=$ROOT/data/bench_u16.bin
    [ -f $alt ] && { AUX="$AUX data_bin=$alt"; echo "  ($tag: $db missing -> $alt)" | tee -a $OUT; }
  fi
  rm -f logs/bench_$tag.log
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag --noprof --wait ${WAIT:-8} --active ${ACTIVE:-12} \
      --batch $B --grad_accum $GA --set $AUX "$@" > logs/bench_$tag.log 2>&1
  rc=$?
  echo "$tag B${B}xGA${GA}${SB:+ sep-backward($SB)} exit=$rc $(grep -E '^\[prof.*tok/s' logs/bench_$tag.log | head -1 | sed 's/^\[prof [a-z_0-9]*\] //') | $(grep -o 'peak allocated.*' logs/bench_$tag.log | tail -1) $(grep -hE 'Error' logs/bench_$tag.log | tail -1 | cut -c1-100)" | tee -a $OUT
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
  grep -qE '^\[prof.*tok/s' logs/bench_$tag.log; }
LATE="mqar_gap_start_frac=0 mqar_gap_ramp_frac=0 mqar_xchunk_start_frac=0"
phase(){ p=$1; cfg=$2; pick=""
  for bg in $3; do b ${p}_xchunk $cfg $bg $LATE mqar_xchunk_frac=1.0 && { pick=$bg; break; }; done
  [ -n "$pick" ] || { echo "$p: no batch split ran -- see logs/bench_${p}_xchunk.log" | tee -a $OUT; return 1; }
  b ${p}_gap $cfg $pick $LATE mqar_xchunk_frac=0
  [ $p = dense ] && b dense_plain $cfg $pick mqar_gap_start_frac=2.0
  echo "$p split: $pick" | tee -a $OUT; eval "PICK_$p=$pick"; }
phase dense $DENSE "${BG:--}"; phase conv $CONV "${BG_CONV:-${BG:--}}"
# EXTRA arms (default on; EXTRA=0 skips): three throughput options on the picked splits, each against its base arm.
# 5090 at 1B (logs/speed1b.log, 10-11): max-autotune +7%, compiled teacher +6% -- and the NorMuon step (Newton-Schulz
# on 60 matrices) is a FIXED ~350 ms of GPU time per optimizer step there, so twice the tokens per step halves its
# cost per token (that one changes the optimisation: half as many steps for the same tokens -- a decision, not a free win).
# To adopt the first two: rerun with SET_ALL="compile_mode=max-autotune-no-cudagraphs" SET_CONV="kd_compile=true
# kd_compile_mode=default" EXTRA=0 for the exact budget, and generate with CLOUD_COMPILE_MODE / CLOUD_KD_COMPILE.
if [ "${EXTRA:-1}" = 1 ] && [ -n "${PICK_dense:-}" ]; then
  tps(){ grep -E '^\[prof.*tok/s' logs/bench_$1.log 2>/dev/null | head -1 | sed -n 's/.*= \([0-9,]*\) tok\/s.*/\1/p' | tr -d ,; }
  gain(){ a=$(tps $1); c=$(tps $2); [ -n "$a" ] && [ -n "$c" ] && echo "  extra: $3 = $a tok/s vs $c ($(awk -v a=$a -v c=$c 'BEGIN{printf "%+.1f%%", (a / c - 1) * 100}'))" | tee -a $OUT; }
  b x_dense_maxauto $DENSE $PICK_dense mqar_gap_start_frac=2.0 compile_mode=max-autotune-no-cudagraphs
  gain x_dense_maxauto dense_plain "dense, compile_mode max-autotune-no-cudagraphs"
  if [ "$PICK_dense" != - ]; then
    IFS=: read -r xb xga xs <<< "$PICK_dense"
    b x_dense_2xtok $DENSE "$xb:$((2 * xga))${xs:+:$xs}" mqar_gap_start_frac=2.0
    gain x_dense_2xtok dense_plain "dense, twice the tokens per optimizer step (B$xb x GA$((2 * xga)))"
  fi
  if [ -n "${PICK_conv:-}" ]; then
    b x_conv_kdcompile $CONV $PICK_conv $LATE mqar_xchunk_frac=0 kd_compile=true kd_compile_mode=default
    gain x_conv_kdcompile conv_gap "conversion, compiled teacher"
  fi
fi
$PY - "$DAYS" "$(key $DENSE mqar_xchunk_frac)" "$(key $DENSE mqar_gap_start_frac)" "${PICK_dense:--}" "${PICK_conv:--}" <<'EOF' | tee -a $OUT
import re, sys
days, xf, gs = float(sys.argv[1]), float(sys.argv[2] or 0.25), float(sys.argv[3] or 0.7)
v = {}
for t in ("dense_plain", "dense_gap", "dense_xchunk", "conv_gap", "conv_xchunk"):
    try:
        m = re.search(r"= ([0-9,]+) tok/s", open(f"logs/bench_{t}.log").read())
    except OSError:
        m = None
    v[t] = float(m.group(1).replace(",", "")) if m else None
if not all(v.values()):
    sys.exit(f"budget: missing arms {[t for t in v if not v[t]]} -- see logs/bench_<arm>.log")
late = lambda g, x: (1 - xf) / v[g] + xf / v[x]                      # seconds per token in the late phase
s_dense = gs / v["dense_plain"] + (1 - gs) * late("dense_gap", "dense_xchunk")
s_conv = late("conv_gap", "conv_xchunk")
r = 0.29 / 0.71
def parts(a):                                                          # "B:GA", "B:GA:s" or "B:GA:sN"
    p = a.split(":"); s = p[2] if len(p) > 2 else ""
    return p[0], p[1], ("" if s[:1] != "s" else (s[1:] if s[1:] not in ("", "1") else " "))   # "" = off, " " = on, "N" = on + N splits
if sys.argv[4] == "-" or sys.argv[5] == "-":
    split = ""
else:
    (dB, dGA, ds), (cB, _, cs) = parts(sys.argv[4]), parts(sys.argv[5])
    sep = ",".join(ph + n.strip() for ph, n in (("d", ds), ("c", cs)) if n) or "-"   # e.g. "d2,c" = dense: separate backward + 2 aux splits; conversion: separate backward
    split = f" {dB} {dGA} {cB} {sep}"
for margin in (0.0, 0.07):                                            # 7% = evals, checkpoints, seed step, restarts
    D = days * 86400 * (1 - margin) / (s_dense + r * s_conv)
    print(f"budget for {days} days, margin {margin:.0%}: dense {D / 1e9:.2f}B + conversion {r * D / 1e9:.2f}B tok "
          f"(mean {1 / s_dense:,.0f} / {1 / s_conv:,.0f} tok/s)  ->  python temp/mk_v4_cloud.py {D:.3e} {r * D:.3e}{split}")
EOF
