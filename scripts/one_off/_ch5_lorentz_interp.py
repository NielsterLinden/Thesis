"""Ch5 5B Lorentz-bias interpretability — checkpoint introspection helpers.

Standalone, no-cfg-build path: reads ``model.pt`` state-dicts directly, slices
out the LorentzScalarBias sub-tree, rebuilds *only* that module, and runs the
end-to-end ``bias(features)`` mapping on a user-supplied feature grid.

Also provides:

- a small "run index" that joins the 60 5B checkpoint directories to the
  axis values in ``thesis_results/04_cleaned_backfilled_analysis_ready.csv``;
- a validation-data replay that returns empirical (5/50/95th-percentile)
  ranges for the seven Lorentz scalar features on the normalized inputs that
  the bias module actually consumes.

Inputs are z-score normalized (see ``H5TokenDataset``); therefore every
feature value in this module is **on the normalized input scale**, not in
raw physics units. Captions on downstream figures must say so explicitly.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

REPO = Path("/project/atlas/users/nterlind/Thesis-Code")
RUNS_DIR = Path("/data/atlas/users/nterlind/outputs/runs")
CSV_PATH = REPO / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv"
DATA_PATH = Path("/data/atlas/users/nterlind/datasets/4tops_splitted.h5")

GROUPS_5B = (
    "exp_20260511-144127_ch5_lorentz_p1",
    "exp_20260511-150429_ch5_lorentz_p2",
)

_LOG = logging.getLogger("ch5_lorentz_interp")
if not _LOG.handlers:
    _LOG.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _LOG.addHandler(_h)


# ---------------------------------------------------------------------------
# Run inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    run_name: str
    run_dir: Path
    features: tuple[str, ...]
    mlp_type: str          # "kan" or "standard"
    sparse_gating: bool
    seed: int
    test_auroc: float


def _parse_features(s: str) -> tuple[str, ...]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ()
    raw = str(s).strip()
    if not raw:
        return ()
    try:
        v = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        v = [t.strip().strip("'\"") for t in raw.strip("[]").split(",")]
    return tuple(v)


def _parse_bool(s) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def load_run_index() -> pd.DataFrame:
    """Load 60-row 5B run inventory with axis columns parsed."""
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["meta_run/group"].isin(GROUPS_5B)].copy()
    if len(df) != 60:
        _LOG.warning("expected 60 5B rows, found %d", len(df))

    df["features"] = df["config/axes/B1-L1_Lorentz Feature Set"].map(_parse_features)
    df["mlp_type"] = df["config/axes/B1-L2_Lorentz MLP Type"].astype(str).str.lower()
    df["sparse_gating"] = df["config/axes/B1-L5_Lorentz Sparse Gating"].map(_parse_bool)
    df["seed"] = df["config/axes/R5_Seed"].astype(int)
    df["run_dir"] = df["meta_run/name"].map(lambda n: RUNS_DIR / str(n))
    df["test_auroc"] = pd.to_numeric(df["eval_v2/test_auroc"], errors="coerce")
    return df.reset_index(drop=True)


def iter_runs(df: pd.DataFrame | None = None) -> list[RunMeta]:
    if df is None:
        df = load_run_index()
    out: list[RunMeta] = []
    for _, r in df.iterrows():
        out.append(
            RunMeta(
                run_id=str(r["meta_run/id"]),
                run_name=str(r["meta_run/name"]),
                run_dir=Path(r["run_dir"]),
                features=r["features"],
                mlp_type=str(r["mlp_type"]),
                sparse_gating=bool(r["sparse_gating"]),
                seed=int(r["seed"]),
                test_auroc=float(r["test_auroc"]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Vendored KAN forward — reconstructs the trained spline mapping
# ---------------------------------------------------------------------------


class _ReconstructedKANLinear(nn.Module):
    """Forward-only KANLinear rebuilt from a state-dict slice."""

    def __init__(
        self,
        base_weight: torch.Tensor,
        spline_weight: torch.Tensor,
        spline_scaler: torch.Tensor,
        grid: torch.Tensor,
        spline_order: int = 3,
        base_activation: type = nn.SiLU,
    ):
        super().__init__()
        self.out_features, self.in_features = base_weight.shape
        self.spline_order = spline_order
        # grid: (in_features, grid_size + 2*spline_order + 1)
        assert grid.shape[0] == self.in_features
        self.grid_size = grid.shape[1] - 1 - 2 * spline_order
        self.register_buffer("grid", grid.clone())
        self.register_buffer("base_weight", base_weight.clone())
        self.register_buffer("spline_weight", spline_weight.clone())
        self.register_buffer("spline_scaler", spline_scaler.clone())
        self.base_activation = base_activation()

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = torch.nn.functional.linear(self.base_activation(x), self.base_weight)
        spline_output = torch.nn.functional.linear(
            self._b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)


class _ReconstructedStandardMLP(nn.Module):
    """Forward-only standard MLP (Linear → SiLU → Linear) from state dict."""

    def __init__(self, w0: torch.Tensor, b0: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
        super().__init__()
        self.fc1 = nn.Linear(w0.shape[1], w0.shape[0])
        self.fc2 = nn.Linear(w2.shape[1], w2.shape[0])
        with torch.no_grad():
            self.fc1.weight.copy_(w0)
            self.fc1.bias.copy_(b0)
            self.fc2.weight.copy_(w2)
            self.fc2.bias.copy_(b2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@dataclass
class LorentzBiasReconstruction:
    mlp: nn.Module                     # callable: (..., F) -> (..., 1)
    gate_raw: float                    # raw scalar (un-tanh'd)
    gate_tanh: float                   # post-tanh scalar — multiplier on bias
    feature_gates_raw: np.ndarray | None
    feature_gates_sigmoid: np.ndarray | None
    features: tuple[str, ...]
    mlp_type: str

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        """End-to-end: applies feature_gates (if any), MLP, and final gate."""
        if self.feature_gates_sigmoid is not None:
            g = torch.tensor(self.feature_gates_sigmoid, dtype=feat.dtype, device=feat.device)
            feat = feat * g
        out = self.mlp(feat).squeeze(-1)
        return out * self.gate_tanh


def load_lorentz_bias(run_dir: Path | str) -> LorentzBiasReconstruction:
    """Rebuild the LorentzScalarBias forward from ``run_dir/model.pt``.

    Reads ``resolved_config.yaml`` for the features list and MLP type, and
    reconstructs either the KAN or standard branch from the state dict.
    """
    run_dir = Path(run_dir)
    ck = torch.load(str(run_dir / "model.pt"), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    cfg_y = yaml.safe_load(open(run_dir / "resolved_config.yaml"))
    lcfg = cfg_y["classifier"]["model"]["bias_config"]["lorentz_scalar"]
    features = tuple(lcfg["features"])
    mlp_type = str(lcfg.get("mlp_type", "standard")).lower()

    prefix = "bias_composer.bias_modules.lorentz_scalar."
    gate_raw = float(sd[prefix + "gate"].item())
    gate_tanh = float(np.tanh(gate_raw))

    fg_key = prefix + "feature_gates"
    if fg_key in sd:
        fg = sd[fg_key].detach().cpu().numpy()
        fg_sigmoid = 1.0 / (1.0 + np.exp(-fg))
    else:
        fg = None
        fg_sigmoid = None

    if mlp_type == "kan":
        layers = []
        for li in (0, 1):
            layers.append(
                _ReconstructedKANLinear(
                    base_weight=sd[f"{prefix}mlp.{li}.base_weight"],
                    spline_weight=sd[f"{prefix}mlp.{li}.spline_weight"],
                    spline_scaler=sd[f"{prefix}mlp.{li}.spline_scaler"],
                    grid=sd[f"{prefix}mlp.{li}.grid"],
                )
            )
        mlp = nn.Sequential(*layers)
    else:
        mlp = _ReconstructedStandardMLP(
            w0=sd[f"{prefix}mlp.0.weight"],
            b0=sd[f"{prefix}mlp.0.bias"],
            w2=sd[f"{prefix}mlp.2.weight"],
            b2=sd[f"{prefix}mlp.2.bias"],
        )
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    return LorentzBiasReconstruction(
        mlp=mlp,
        gate_raw=gate_raw,
        gate_tanh=gate_tanh,
        feature_gates_raw=fg,
        feature_gates_sigmoid=fg_sigmoid,
        features=features,
        mlp_type=mlp_type,
    )


# ---------------------------------------------------------------------------
# Validation-data replay: empirical feature ranges on normalized inputs
# ---------------------------------------------------------------------------


def replay_validation_features(
    n_events: int = 4000,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Compute pairwise Lorentz features on a normalized validation sample.

    Returns a dict {feature_name: 1D array of off-diagonal valid pair values},
    suitable for empirical-range and histogram plots.
    """
    import sys

    sys.path.insert(0, str(REPO / "src"))
    import h5py

    from thesis_ml.architectures.transformer_classifier.modules.biases._features import (
        compute_pairwise_feature_set,
    )

    # Mirror H5TokenDataset normalization: z-score using TRAIN stats.
    with h5py.File(DATA_PATH, "r") as f:
        Xtr = f["X_train"][:200_000]
        Xva = f["X_val"][:]
    T = 18
    cont_features = [0, 1, 2, 3]
    train_cont = torch.tensor(Xtr[:, T + 2 :], dtype=torch.float32).view(-1, T, 4)
    val_cont = torch.tensor(Xva[:, T + 2 :], dtype=torch.float32).view(-1, T, 4)
    val_ids = torch.tensor(Xva[:, :T], dtype=torch.long)
    mu = train_cont.mean(dim=(0, 1), keepdim=True)
    sd = train_cont.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
    val_norm = (val_cont - mu) / sd  # [N, T, 4]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val_norm), size=min(n_events, len(val_norm)), replace=False)
    val_norm = val_norm[idx].to(device)
    val_ids = val_ids[idx].to(device)

    mask = val_ids != 0  # [N, T]

    feat, names = compute_pairwise_feature_set(
        val_norm,
        list(["m2", "dot", "log_m2", "log_kt", "z", "deltaR", "deltaR_ptw"]),
        mask=mask,
    )
    # feat: [N, T, T, F]; build off-diagonal valid mask
    N, Tt, _, F = feat.shape
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    iu = torch.triu_indices(Tt, Tt, offset=1)
    pair_mask = pair_mask[:, iu[0], iu[1]]              # [N, P]
    feat = feat[:, iu[0], iu[1], :]                     # [N, P, F]
    out: dict[str, np.ndarray] = {}
    for k, name in enumerate(names):
        v = feat[:, :, k][pair_mask].detach().cpu().numpy()
        out[name] = v
    return out


def empirical_range(values: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """Return (loth, hith) percentile cutoffs as a robust plotting domain."""
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


# ---------------------------------------------------------------------------
# Single-feature partial-dependence sweep helper
# ---------------------------------------------------------------------------


def sweep_1d(
    recon: LorentzBiasReconstruction,
    feature_name: str,
    x: np.ndarray,
    baseline: dict[str, float] | None = None,
) -> np.ndarray:
    """Evaluate ``bias(x)`` along one feature with the others held at baseline.

    Parameters
    ----------
    recon : LorentzBiasReconstruction
        Loaded module.
    feature_name : str
        Feature on which to sweep. Must be in ``recon.features``.
    x : np.ndarray
        1-D values for the swept feature (on the normalized scale).
    baseline : dict[str, float] | None
        Values for the *non-swept* features. Missing keys default to 0.

    Returns
    -------
    np.ndarray
        End-to-end bias values, shape ``x.shape``.
    """
    if feature_name not in recon.features:
        raise KeyError(f"{feature_name} not in {recon.features}")
    F = len(recon.features)
    baseline = baseline or {}
    base = torch.zeros(F, dtype=torch.float32)
    for j, fn in enumerate(recon.features):
        if fn != feature_name and fn in baseline:
            base[j] = float(baseline[fn])
    sweep_idx = recon.features.index(feature_name)
    xt = torch.tensor(x, dtype=torch.float32)
    feat = base.unsqueeze(0).expand(xt.shape[0], -1).clone()
    feat[:, sweep_idx] = xt
    with torch.no_grad():
        out = recon(feat)
    return out.detach().cpu().numpy()


def sweep_2d(
    recon: LorentzBiasReconstruction,
    feat_x: str,
    feat_y: str,
    xs: np.ndarray,
    ys: np.ndarray,
    baseline: dict[str, float] | None = None,
) -> np.ndarray:
    """Evaluate ``bias(x, y)`` on a 2-D grid, others held at baseline."""
    F = len(recon.features)
    base = torch.zeros(F, dtype=torch.float32)
    baseline = baseline or {}
    for j, fn in enumerate(recon.features):
        if fn not in (feat_x, feat_y) and fn in baseline:
            base[j] = float(baseline[fn])
    ix = recon.features.index(feat_x)
    iy = recon.features.index(feat_y)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    flat = base.unsqueeze(0).expand(X.size, -1).clone()
    flat[:, ix] = torch.tensor(X.reshape(-1), dtype=torch.float32)
    flat[:, iy] = torch.tensor(Y.reshape(-1), dtype=torch.float32)
    with torch.no_grad():
        out = recon(flat)
    return out.detach().cpu().numpy().reshape(X.shape)
