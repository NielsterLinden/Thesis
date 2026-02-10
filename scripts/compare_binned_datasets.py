#!/usr/bin/env python
"""Compare our binned dataset with Ambre's pre-binned dataset.

Run interactively on HPC:
  python scripts/compare_binned_datasets.py \\
    --ours /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \\
    --ambres /data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _inspect_h5(path: Path) -> dict:
    """Inspect H5 file structure and return keys, shapes, dtypes."""
    info = {"path": str(path), "keys": {}, "attrs": {}}
    with h5py.File(path, "r") as f:
        for key in f:
            d = f[key]
            if isinstance(d, h5py.Dataset):
                info["keys"][key] = {"shape": d.shape, "dtype": str(d.dtype)}
            else:
                info["keys"][key] = "group"
        for key, val in f.attrs.items():
            info["attrs"][key] = val
    return info


def _load_dataset(f: h5py.File, key: str):
    """Load dataset, trying common key variants."""
    if key in f:
        return f[key][:]
    # Try alternate casing
    for k in f:
        if k.lower() == key.lower():
            return f[k][:]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare our binned dataset with Ambre's.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ours",
        type=Path,
        required=True,
        help="Path to our binned H5 file.",
    )
    parser.add_argument(
        "--ambres",
        type=Path,
        required=True,
        help="Path to Ambre's binned H5 file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Max samples to compare (for speed).",
    )
    args = parser.parse_args()

    ours_path = args.ours.resolve()
    ambres_path = args.ambres.resolve()

    if not ours_path.exists():
        print(f"Error: our file not found: {ours_path}")
        sys.exit(1)
    if not ambres_path.exists():
        print(f"Error: Ambre's file not found: {ambres_path}")
        sys.exit(1)

    print("=" * 60)
    print("Inspection: Our binned dataset")
    print("=" * 60)
    ours_info = _inspect_h5(ours_path)
    for k, v in ours_info["keys"].items():
        print(f"  {k}: {v}")
    for k, v in ours_info["attrs"].items():
        print(f"  @{k}: {v}")

    print()
    print("=" * 60)
    print("Inspection: Ambre's binned dataset")
    print("=" * 60)
    ambres_info = _inspect_h5(ambres_path)
    for k, v in ambres_info["keys"].items():
        print(f"  {k}: {v}")
    for k, v in ambres_info["attrs"].items():
        print(f"  @{k}: {v}")

    # Load our data
    print()
    print("=" * 60)
    print("Loading and comparing")
    print("=" * 60)

    with h5py.File(ours_path, "r") as f:
        Xtr_ours = _load_dataset(f, "X_train")
        if Xtr_ours is None:
            print("Could not find X_train in our file")
            sys.exit(1)
        Xtr_ours = np.asarray(Xtr_ours)

    # Ambre's file may have different structure (train_AND_test suggests combined split)
    with h5py.File(ambres_path, "r") as f:
        # Try common keys
        X_ambres = None
        for key in ["X_train", "X_train_and_test", "X", "tokens", "data"]:
            data = _load_dataset(f, key)
            if data is not None and hasattr(data, "shape"):
                X_ambres = np.asarray(data)
                print(f"  Ambre's data from key '{key}': shape {X_ambres.shape}")
                break
        if X_ambres is None:
            # Use first dataset that looks like token data
            for key in f:
                d = f[key]
                if isinstance(d, h5py.Dataset) and d.ndim >= 2:
                    X_ambres = d[:]
                    print(f"  Ambre's data from key '{key}': shape {X_ambres.shape}")
                    break
        if X_ambres is None:
            print("Could not find token data in Ambre's file")
            sys.exit(1)

    n_ours = min(args.n_samples, len(Xtr_ours))
    n_ambres = min(args.n_samples, len(X_ambres))

    Xo = Xtr_ours[:n_ours]
    Xa = X_ambres[:n_ambres]

    # Compare shapes
    print(f"\n  Our shape: {Xo.shape}")
    print(f"  Ambre shape: {Xa.shape}")

    # Compare vocab/range
    print(f"\n  Our token range: [{Xo.min()}, {Xo.max()}]")
    print(f"  Ambre token range: [{Xa.min()}, {Xa.max()}]")

    # Compare column layout (if same number of columns)
    if Xo.shape[1] == Xa.shape[1]:
        print("\n  Column-wise token value ranges (ours vs Ambre):")
        for c in range(min(5, Xo.shape[1])):  # First 5 columns
            print(f"    col {c}: ours [{Xo[:, c].min()}, {Xo[:, c].max()}]  " f"Ambre [{Xa[:, c].min()}, {Xa[:, c].max()}]")
        if Xo.shape[1] > 5:
            print(f"    ... ({Xo.shape[1]} columns total)")
    else:
        print(f"\n  Column count differs: ours={Xo.shape[1]}, Ambre={Xa.shape[1]}")

    # If we have matching event count and same structure, compute agreement
    if Xo.shape[0] == Xa.shape[0] and Xo.shape[1] == Xa.shape[1]:
        match = (Xo == Xa).mean() * 100
        print(f"\n  Token-by-token agreement: {match:.2f}%")
    else:
        print("\n  (Skipping agreement: different shapes or sample sizes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
