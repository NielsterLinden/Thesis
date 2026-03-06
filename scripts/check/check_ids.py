import logging
from pathlib import Path

import h5py
import numpy as np
from dotenv import load_dotenv
from hydra import compose, initialize

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        load_dotenv()
        initialize(config_path=str(CONFIGS_DIR), version_base="1.3")
        cfg = compose(config_name="config")
    except Exception as e:
        logger.error("Failed to compose Hydra config: %s", e)
        return 1

    # Read H5 directly to avoid torch dependency for this quick check
    path = cfg.data.path
    try:
        with h5py.File(path, "r") as f:
            splits = {
                "train": f[cfg.data.datasets.x_train][:],
                "val": f[cfg.data.datasets.x_val][:],
                "test": f[cfg.data.datasets.x_test][:],
            }
    except Exception as e:
        logger.error("Failed to open H5 file. Ensure DATA_ROOT/DATA_FILE envs are set, as used in configs/data/h5_dataset.yaml.")
        logger.error("Path resolved to: %s", path)
        logger.error("Original error: %s", e)
        return 2

    T = int(cfg.data.n_tokens)

    def min_max_ids(X):
        ids = X[:, :T].astype(np.int64).ravel()
        return int(ids.min()), int(ids.max())

    mins = []
    maxs = []
    for name, X in splits.items():
        mn, mx = min_max_ids(X)
        mins.append(mn)
        maxs.append(mx)
        logger.info("%s: min_id=%s, max_id=%s", name, mn, mx)
    logger.info("overall: min_id=%s, max_id=%s", min(mins), max(maxs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
