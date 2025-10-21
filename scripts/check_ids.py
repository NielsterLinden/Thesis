import h5py
import numpy as np
from dotenv import load_dotenv
from hydra import compose, initialize


def main() -> int:
    try:
        load_dotenv()
        # Load the same Hydra config as training without changing cwd
        initialize(config_path="../configs", version_base="1.3")
        cfg = compose(config_name="config")
    except Exception as e:
        print(f"Failed to compose Hydra config: {e}")
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
        print("Failed to open H5 file. Ensure DATA_ROOT/DATA_FILE envs are set, as used in configs/data/h5_dataset.yaml.")
        print(f"Path resolved to: {path}")
        print(f"Original error: {e}")
        return 2

    T = int(cfg.data.n_tokens)

    def min_max_ids(X: np.ndarray):
        ids = X[:, :T].astype(np.int64).ravel()
        return int(ids.min()), int(ids.max())

    mins = []
    maxs = []
    for name, X in splits.items():
        mn, mx = min_max_ids(X)
        mins.append(mn)
        maxs.append(mx)
        print(f"{name}: min_id={mn}, max_id={mx}")
    print(f"overall: min_id={min(mins)}, max_id={max(maxs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
