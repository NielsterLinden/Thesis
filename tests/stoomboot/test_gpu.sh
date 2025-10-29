#!/usr/bin/env bash
set -eo pipefail            # no -u here (or do: set +u before activate)

echo "=== GPU test on $(hostname) ==="
source /data/atlas/users/nterlind/venvs/miniconda3/etc/profile.d/conda.sh
# if you keep -u elsewhere, you can do: set +u; conda activate ...; set -u
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

python - <<'PY'
import torch, json
print(json.dumps({
  "torch": torch.__version__,
  "cuda_built": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY
