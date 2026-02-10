"""Ambre-style binning for particle physics tokenization.

Converts continuous features (pT, eta, phi) and object IDs into discrete integer
tokens suitable for embedding-based models. Uses quantile-based binning for pT,
eta, and MET; fixed-width binning for phi and MET phi.
"""

from __future__ import annotations

import numpy as np
import torch


class AmbreBinning:
    """Ambre-style binning: map (cont, ids, globals) → integer tokens [1, vocab_size-1].

    Formula for 5 bins, 7 IDs:
    - token_part = 125*(bin_obj−1) + 25*(bin_pT−1) + 5*(bin_η−1) + bin_φ ∈ [1, 875]
    - MET: 5 bins → [876, 880]
    - MET phi: 5 bins → [881, 885]
    - Padding: 0
    """

    def __init__(self, n_bins: int = 5, n_ids: int = 7):
        """Initialize binning (edges computed in fit).

        Parameters
        ----------
        n_bins : int
            Number of bins for pT, eta, phi, MET, MET phi.
        n_ids : int
            Number of object ID types (jet, b-jet, e+, e-, mu+, mu-, photon).
        """
        self.n_bins = n_bins
        self.n_ids = n_ids
        self.n_part_tokens = n_ids * n_bins * n_bins * n_bins  # 7*5*5*5 = 875
        self.met_offset = self.n_part_tokens + 1  # 876
        self.met_phi_offset = self.met_offset + n_bins  # 881
        self.vocab_size = 1 + self.n_part_tokens + n_bins + n_bins  # 0 + 875 + 5 + 5 = 886

        # Bin edges (set in fit)
        # cont layout: [E, pT, eta, phi] per token; indices: 0=E, 1=pT, 2=eta, 3=phi
        self.edges_pT: np.ndarray | None = None
        self.edges_eta: np.ndarray | None = None
        self.edges_met: np.ndarray | None = None
        # phi, met_phi: fixed width π/n_bins
        self._phi_width = np.pi / n_bins
        self._met_phi_width = np.pi / n_bins
        self._fitted = False

    def fit(
        self,
        train_cont: np.ndarray | torch.Tensor,
        train_ids: np.ndarray | torch.Tensor,
        train_globals: np.ndarray | torch.Tensor,
        *,
        use_background_only: bool = False,
        bg_labels: list[int] | None = None,
        train_labels: np.ndarray | torch.Tensor | None = None,
    ) -> AmbreBinning:
        """Compute bin edges from training data.

        Parameters
        ----------
        train_cont : np.ndarray or torch.Tensor
            [N, T, 4] continuous features (E, pT, eta, phi).
        train_ids : np.ndarray or torch.Tensor
            [N, T] object IDs.
        train_globals : np.ndarray or torch.Tensor
            [N, 2] (MET, MET phi).
        use_background_only : bool
            If True, compute quantiles only from background events.
        bg_labels : list[int]
            Background labels when use_background_only is True.
        train_labels : np.ndarray or torch.Tensor
            [N] labels for filtering background.

        Returns
        -------
        AmbreBinning
            self (for chaining).
        """
        if isinstance(train_cont, torch.Tensor):
            train_cont = train_cont.numpy()
        if isinstance(train_ids, torch.Tensor):
            train_ids = train_ids.numpy()
        if isinstance(train_globals, torch.Tensor):
            train_globals = train_globals.numpy()

        if use_background_only and train_labels is not None and bg_labels is not None:
            if isinstance(train_labels, torch.Tensor):
                train_labels = train_labels.numpy()
            mask = np.isin(train_labels, bg_labels)
            train_cont = train_cont[mask]
            train_ids = train_ids[mask]
            train_globals = train_globals[mask]

        # Flatten to [N*T] for quantiles (only valid tokens: id != 0)
        N, T, _ = train_cont.shape
        cont_flat = train_cont.reshape(-1, 4)
        ids_flat = train_ids.reshape(-1)
        valid = ids_flat != 0
        cont_valid = cont_flat[valid]

        # pT (index 1), eta (index 2)
        pT_vals = cont_valid[:, 1]
        eta_vals = cont_valid[:, 2]
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.edges_pT = np.percentile(pT_vals, quantiles).astype(np.float32)
        self.edges_eta = np.percentile(eta_vals, quantiles).astype(np.float32)

        # MET (globals index 0)
        met_vals = train_globals[:, 0]
        met_valid = met_vals[met_vals > 0]  # avoid zeros
        if len(met_valid) > 0:
            self.edges_met = np.percentile(met_valid, quantiles).astype(np.float32)
        else:
            self.edges_met = np.linspace(0, met_vals.max(), self.n_bins + 1)[1:-1].astype(np.float32)

        self._fitted = True
        return self

    def _digitize_quantile(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Map values to bin indices 1..n_bins. Values outside range go to 1 or n_bins."""
        out = np.digitize(x, edges)  # 0..n_bins
        out = np.clip(out, 1, self.n_bins)  # 1-indexed [1, n_bins]
        return out.astype(np.int64)

    def _digitize_phi(self, phi: np.ndarray) -> np.ndarray:
        """Map phi ∈ [-π, π] to bin indices 1..n_bins (fixed width π/n_bins)."""
        # phi in [-π, π], shift to [0, 2π] for binning
        phi_shifted = np.mod(phi + np.pi, 2 * np.pi)
        bin_idx = np.floor(phi_shifted / self._phi_width).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1) + 1  # 1-indexed
        return bin_idx

    def transform(
        self,
        cont: np.ndarray | torch.Tensor,
        ids: np.ndarray | torch.Tensor,
        globals_: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Transform raw events to integer tokens.

        Parameters
        ----------
        cont : np.ndarray or torch.Tensor
            [N, T, 4] continuous (E, pT, eta, phi).
        ids : np.ndarray or torch.Tensor
            [N, T] object IDs.
        globals_ : np.ndarray or torch.Tensor
            [N, 2] (MET, MET phi).

        Returns
        -------
        np.ndarray
            [N, T+2] integer tokens (18 particles + MET + MET phi).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        if isinstance(cont, torch.Tensor):
            cont = cont.numpy()
        if isinstance(ids, torch.Tensor):
            ids = ids.numpy()
        if isinstance(globals_, torch.Tensor):
            globals_ = globals_.numpy()

        N, T, _ = cont.shape
        out = np.zeros((N, T + 2), dtype=np.int64)

        pT = cont[:, :, 1]
        eta = cont[:, :, 2]
        phi = cont[:, :, 3]
        obj = np.clip(ids, 1, self.n_ids)  # 1-indexed, clip to valid range

        bin_pT = self._digitize_quantile(pT, self.edges_pT)
        bin_eta = self._digitize_quantile(eta, self.edges_eta)
        bin_phi = self._digitize_phi(phi)

        # token_part = 125*(bin_obj−1) + 25*(bin_pT−1) + 5*(bin_eta−1) + bin_phi
        stride = self.n_bins * self.n_bins * self.n_bins  # 125
        token_part = (obj - 1) * stride + (bin_pT - 1) * self.n_bins * self.n_bins + (bin_eta - 1) * self.n_bins + bin_phi
        # Padding: id=0 -> token=0
        token_part = np.where(ids != 0, token_part, 0)
        out[:, :T] = token_part

        # MET and MET phi
        met = globals_[:, 0]
        met_phi = globals_[:, 1]
        bin_met = self._digitize_quantile(met, self.edges_met)
        bin_met_phi = self._digitize_phi(met_phi)
        out[:, T] = self.met_offset + (bin_met - 1)
        out[:, T + 1] = self.met_phi_offset + (bin_met_phi - 1)

        return out
