#!/usr/bin/env bash
set -euo pipefail

echo "=== GPU test on $(hostname) ==="
source /data/atlas/users/nterlind/venvs/miniconda3/etc/profile.d/conda.sh
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

python - <<'PY'
import torch, json
info = {
    "torch": torch.__version__,
    "cuda_built": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(info, indent=2))
PY
