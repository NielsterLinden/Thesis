#!/usr/bin/env bash
set -Eeuxo pipefail

echo "=== GPU test on $(hostname) ==="
echo "PWD=$(pwd)"
which conda || true
conda --version || true

# Load conda in non-interactive shells
source /data/atlas/users/nterlind/venvs/miniconda3/etc/profile.d/conda.sh

# Activate env
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

which python
python -V
nvidia-smi || true

python - <<'PY'
import torch, json, sys
info = {
    "torch": getattr(torch, "__version__", None),
    "cuda_built": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(info, indent=2))
sys.exit(0)
PY
