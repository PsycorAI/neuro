#!/bin/bash
# setup_instance.sh -- run ONCE on the Vast instance (2026-10-04 draft). Builds a venv with the same versions as the
# home 5090 stack (torch 2.11.0+cu128, triton 3.6.0, liger 0.8.0, fla 0.5.1) and smoke-tests the GPU.
set -e
# train_gpu hardcodes /home/glenn/projects/neuro/{models/checkpoints,data,models/brains}: mirror that path to /workspace.
mkdir -p /home/glenn/projects && ln -sfn /workspace/neuro /home/glenn/projects/neuro
cd /workspace/neuro
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install -q --upgrade pip
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
/workspace/venv/bin/pip install -q triton==3.6.0 numpy==2.4.5 PyYAML==6.0.3 liger_kernel==0.8.0 einops==0.8.2 \
    fla-core==0.5.1 flash-linear-attention==0.5.1 safetensors==0.7.0
# model.py / train_gpu imports (missed in the draft; first H200 run 2026-10-04)
/workspace/venv/bin/pip install -q snntorch==0.9.4 transformers==5.8.1 datasets==4.8.5 tqdm==4.67.3 tokenizers==0.22.2
(cd src && /workspace/venv/bin/python -c "import train_gpu; print('train_gpu import ok')")
/workspace/venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).float().abs().mean().item() > 0)
EOF
nproc; lscpu | grep -E 'Model name|MHz' | head -3
echo "[setup] done"
