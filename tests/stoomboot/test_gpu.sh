#!/usr/bin/env bash
# keep it simple; don't use -u during conda activation
set -eo pipefail

echo "=== GPU test on $(hostname) ==="

# allow unset vars while sourcing conda
set +u
source /data/atlas/users/nterlind/venvs/miniconda3/etc/profile.d/conda.sh
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u || true

python - <<'PY'
import torch, json
print(json.dumps({
  "torch": torch.__version__,
  "cuda_built": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY
