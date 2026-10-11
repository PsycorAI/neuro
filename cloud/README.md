# v4 cloud run on a Vast.ai H200 (interruptible)

Rewritten 2026-10-11 (Neuro 55) for the RC3 two-phase run. The 10-04 version described the old layout and a 46 GB rsync.

The run has two phases, both driven by `cloud/run_v4.sh`:

1. **Dense** (`configs/v4_cloud.yaml`): 12 attention layers + the spiking memory, on the uint16 corpus.
2. **Conversion** (`configs/v4_cloud_conv.yaml`): spike codes + Llama-3.2-1B distillation, on the raw uint32 KD slice. `temp/mk_spike_seed.py` seeds it from the last dense checkpoint, in its own run directory.

Interruptible means the instance can stop whenever someone outbids us. `run_v4.sh` is the on-start command, so the run resumes from its latest checkpoint when the instance comes back.

## What has and has not been proven at home (2026-10-11)

| Piece | State |
|---|---|
| Dense phase at 1B, T2048, with a stop + resume and the gap phase | Passed on the 5090 (B1) |
| Seed step at 1B | Passed |
| Conversion at 1B | Does not fit the 5090 at T2048; tested at T1024 instead (see `docs/to-do.html`, 10-11) |
| Whole pipeline end to end | Passed at 7L on the 5080 (10-10 rehearsal) |
| `pull_corpus.sh` (gdown from Drive) | **Never run** |
| `pull_ckpts.sh`, `cloud_watch.sh`, `get_teacher.sh`, `bench_rc3.sh` on a real instance | **Never run** (logic tested at home where possible) |
| B16 at T2048 on 143 GB | **Unknown** — `bench_rc3.sh` finds the largest batch split that fits |

## Steps

1. **Rent:** 1× H200, interruptible, PyTorch CUDA 12.8 image, **at least 300 GB of disk** (corpus pieces 46 GB + joined file 46 GB + KD slice 2 × 8 GB + checkpoints ~10 GB each), a fast host CPU. On-start command: `bash /workspace/neuro/cloud/run_v4.sh` (it does nothing until the `GO` file exists).
2. **Push from home (WSL):** `bash cloud/push_from_home.sh <host> <port>` — code, configs, eval scripts, vocab map. A few MB.
3. **On the instance, once:**
   - `bash /workspace/neuro/cloud/setup_instance.sh` — venv + GPU check.
   - `bash /workspace/neuro/cloud/pull_corpus.sh` — downloads whatever corpus pieces are on Drive and joins them. Re-run it when more pieces are up, then restart the trainer (it sees the file size only at start).
   - `HF_TOKEN=hf_... bash /workspace/neuro/cloud/get_teacher.sh` — the gated teacher. Without a token on the instance: `TEACHER=1 bash cloud/push_from_home.sh <host> <port>` from home (~4 h), then `bash cloud/get_teacher.sh` to check it.
4. **Bench (30–45 min):** `BG="16:1:s2 8:2:s 4:4:s" bash /workspace/neuro/cloud/bench_rc3.sh`. It prints tok/s and peak memory per kind of step, the batch split that fits, the token budget for 4.5 days, and the exact `mk_v4_cloud.py` command. `s` = `mqar_separate_backward` (the main loss is back-propagated before the recall batch, so their memory does not add up); `s2` also runs the recall batch in two halves, which batch 16 needs. Estimated peaks on 143 GB: `16:1:s2` ~85 GB, `8:2:s` ~77 GB dense / ~84 GB conversion.
5. **Set the budget at home:** run the printed `python temp/mk_v4_cloud.py <DENSE> <CONV> <B> <GA> <CONV_B> <SEP>` (SEP looks like `d2,c`), run `/audit-config`, then push again (step 2).
6. **Launch on the instance:** `touch /workspace/neuro/GO && cd /workspace/neuro && setsid -f bash cloud/run_v4.sh < /dev/null > /dev/null 2>&1`
7. **Off-instance checkpoint copies (on the instance):** `setsid -f bash cloud/push_ckpts_drive.sh` — newest checkpoint of each phase to Google Drive every 6 h with rclone, at datacenter speed (the home line is 19 Mbps = 70+ min per 10 GB checkpoint). Needs a one-time `rclone config` by Glenn; the steps are in the script header. Never run yet.
8. **At home, under `setsid -f`:**
   - (optional, instead of step 7) `bash cloud/pull_ckpts.sh <host> <port>` — newest checkpoint of both phases every 2 h over the home line, keeps the last 2 per phase.
   - `bash cloud/cloud_watch.sh <host> <port>` — alerts (toast + `logs/alerts.log`) if the instance is unreachable, the run stops, the tripwire fires, the log goes silent, or the disk fills; a status line every 6 h. Set `MIN_TPS` to about 60% of the benched speed.

## Things to know

- **Stops that do not retry:** the val_ppl tripwire (exit 2) and the NaN guard (exit 3) stop `run_v4.sh` for a human. The instance keeps billing while idle; `cloud_watch.sh` is what tells us.
- **Memory go/no-go:** RC3 trains the memory from 70% of the dense phase. Check on the instance: `PY=/workspace/venv/bin/python T=2048 bash temp/run_r25_evals.sh v4_cloud=<checkpoint>`.
- **State from one instance to the next:** everything lives under `/workspace`. If the host is lost for good, the newest pulled checkpoint at home is the restart point.
- **Status by hand:** `bash /workspace/neuro/cloud/status.sh`.
