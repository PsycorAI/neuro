"""mk_bench_h200.py -- 2026-10-04: H200 throughput-bench inputs. Glenn's DSL upload is 1.3 Mbps, so the bench uses
an 8M-token slice of the uint16 corpus (16 MB) instead of the 46 GB file (data content does not change step cost).
Config = probe_g4_r19d (1B) at Glenn's 5% sparse-MLP target + x1 gap MQAR switched on from step 0 (worst case:
gap batches every step), data_prelut on the slice."""
import yaml, numpy as np
R = "/home/glenn/projects/neuro/"
SRC = "/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin"
DST = R + "temp/bench_slice_u16.bin"
src = np.memmap(SRC, dtype=np.uint16, mode="r")
np.asarray(src[:8_000_000]).tofile(DST)
c = yaml.safe_load(open(R + "configs/probe_g4_r19d.yaml"))
c.update(run_name="v4_cloud_bench", attn_ffn_rate_target=0.05,
         mqar_gap_max=440, mqar_gap_start_frac=0.0, mqar_gap_ramp_frac=0.0, mqar_gap_weight=0.15, mqar_batch=8,
         mqar_pad_to=65, mqar_gap_pad_to=512, mqar_compile_static=True, gdn2_cumsum_eager_max_b=8,
         data_bin="/workspace/neuro/temp/bench_slice_u16.bin", data_prelut=True, data_val_tokens=1_000_000)
with open(R + "configs/v4_cloud_bench.yaml", "w") as f:
    f.write("# v4_cloud_bench -- AUTO-GEN temp/mk_bench_h200.py (2026-10-04). H200 throughput bench only.\n")
    yaml.safe_dump(c, f, sort_keys=False)
print("[OK] slice", DST, "+ configs/v4_cloud_bench.yaml")
