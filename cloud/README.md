# v4 cloud run on a Vast.ai H200 (interruptible)

This directory was drafted on 2026-10-04. Nothing has run yet: we're waiting on Glenn's Vast account.

Choosing interruptible means the run can stop whenever someone outbids us. It resumes from the latest checkpoint once our bid wins again. The run is set up to survive this:

- Checkpoints are saved often (`ckpt_every`, about every 20–30 minutes).
- `run_v4.sh` restarts training from the latest checkpoint on every start, so it doubles as the Vast on-start command.
- `pull_ckpts.sh` runs at home and copies the newest checkpoint back about every 2 hours. That way, losing the host loses at most ~2 h.

## Steps

1. **Glenn:** create the Vast account, add credit, and register the SSH public key from `~/.ssh/id_ed25519.pub` (WSL).
2. **Rent the instance:** 1× H200, interruptible. Use the PyTorch CUDA 12.8 image, at least 200 GB of disk, and a fast host CPU, since the run is launch-bound. Set `bash /workspace/neuro/cloud/run_v4.sh` as the on-start command.
3. **Push from home:** `bash cloud/push_from_home.sh <ssh-host> <ssh-port>`. This copies the code and the uint16 corpus (`train_lut32768_u16.bin`, 46 GB) plus the vocab map.
4. **On the instance, once:** `bash /workspace/neuro/cloud/setup_instance.sh`. This builds the venv and runs a GPU smoke test.
5. **Benchmark (about 1 hour):** `bash /workspace/neuro/cloud/bench_h200.sh`. It measures tok/s for B32×GA2 and B16×GA4, which sets the token budget that fits the Oct 19 deadline.
6. **Launch:** `bash /workspace/neuro/cloud/run_v4.sh`, which loops and auto-resumes. At home, run `bash cloud/pull_ckpts.sh <ssh-host> <ssh-port>` under `setsid`.

The config `configs/v4_cloud.yaml` gets frozen after rounds r21/r22. Run `/audit-config` on it before launch.
