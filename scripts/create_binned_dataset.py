#!/usr/bin/env python
"""Create Ambre-style binned H5 dataset from raw 4tops data.

Run interactively on HPC:
  python scripts/create_binned_dataset.py \\
    --input /data/atlas/users/nterlind/datasets/4tops_splitted.h5 \\
    --output /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \\
    --n-bins 5
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

from thesis_ml.data.binning import AmbreBinning  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Create Ambre-style binned H5 from raw 4tops data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw H5 file (4tops_splitted.h5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output binned H5 file.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of bins for pT, eta, phi, MET, MET phi.",
    )
    parser.add_argument(
        "--x-train",
        type=str,
        default="X_train",
        help="HDF5 key for training data.",
    )
    parser.add_argument(
        "--x-val",
        type=str,
        default="X_val",
        help="HDF5 key for validation data.",
    )
    parser.add_argument(
        "--x-test",
        type=str,
        default="X_test",
        help="HDF5 key for test data.",
    )
    parser.add_argument(
        "--y-train",
        type=str,
        default="Y_train",
        help="HDF5 key for training labels.",
    )
    parser.add_argument(
        "--y-val",
        type=str,
        default="y_val",
        help="HDF5 key for validation labels.",
    )
    parser.add_argument(
        "--y-test",
        type=str,
        default="y_test",
        help="HDF5 key for test labels.",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=18,
        help="Number of particle tokens per event.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = args.n_tokens

    print(f"Loading raw data from {input_path}...")
    with h5py.File(input_path, "r") as f:
        Xtr = f[args.x_train][:]
        Xva = f[args.x_val][:]
        Xte = f[args.x_test][:]

        Ytr = f.get(args.y_train, None)
        Yva = f.get(args.y_val, None)
        Yte = f.get(args.y_test, None)
        if Ytr is None:
            raise ValueError(f"Labels {args.y_train} not found in H5 file")
        Ytr = Ytr[:]
        Yva = Yva[:] if Yva is not None else Ytr[: len(Xva)]
        Yte = Yte[:] if Yte is not None else Ytr[: len(Xte)]

    print(f"  Train: {len(Xtr)} events, Val: {len(Xva)}, Test: {len(Xte)}")

    cont_tr = Xtr[:, T + 2 :].reshape(-1, T, 4)
    ids_tr = Xtr[:, :T].astype(np.int64)
    gl_tr = Xtr[:, T : T + 2]

    binning = AmbreBinning(n_bins=args.n_bins, n_ids=7)
    print(f"Fitting binning (n_bins={args.n_bins}) on training data...")
    binning.fit(cont_tr, ids_tr, gl_tr)

    print("Transforming splits...")
    tokens_tr = binning.transform(cont_tr, ids_tr, gl_tr)
    cont_va = Xva[:, T + 2 :].reshape(-1, T, 4)
    ids_va = Xva[:, :T].astype(np.int64)
    gl_va = Xva[:, T : T + 2]
    tokens_va = binning.transform(cont_va, ids_va, gl_va)
    cont_te = Xte[:, T + 2 :].reshape(-1, T, 4)
    ids_te = Xte[:, :T].astype(np.int64)
    gl_te = Xte[:, T : T + 2]
    tokens_te = binning.transform(cont_te, ids_te, gl_te)

    print(f"  Token range: [0, {binning.vocab_size - 1}]")
    print(f"  Output shape: [N, {T + 2}] (18 particles + MET + MET phi)")

    print(f"Writing binned data to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("X_train", data=tokens_tr, dtype=np.int64, compression="gzip")
        f.create_dataset("X_val", data=tokens_va, dtype=np.int64, compression="gzip")
        f.create_dataset("X_test", data=tokens_te, dtype=np.int64, compression="gzip")
        f.create_dataset("Y_train", data=Ytr, dtype=np.float32)
        f.create_dataset("y_val", data=Yva, dtype=np.float32)
        f.create_dataset("y_test", data=Yte, dtype=np.float32)
        f.attrs["n_bins"] = args.n_bins
        f.attrs["vocab_size"] = binning.vocab_size
        f.attrs["n_tokens"] = T

    print("Done.")


if __name__ == "__main__":
    main()
