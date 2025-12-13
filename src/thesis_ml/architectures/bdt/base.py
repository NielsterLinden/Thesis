"""BDT Classifier wrapper around XGBoost."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from omegaconf import DictConfig

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("XGBoost is required for BDT classifier. Install with: conda install xgboost") from e


class BDTClassifier:
    """XGBoost-based Boosted Decision Tree classifier.

    This is a wrapper around XGBClassifier that provides a consistent
    interface with the PyTorch-based classifiers.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_classes: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
    ):
        """Initialize BDT classifier.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds (trees)
        max_depth : int
            Maximum depth of each tree
        learning_rate : float
            Boosting learning rate (eta)
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree
        min_child_weight : int
            Minimum sum of instance weight needed in a child
        gamma : float
            Minimum loss reduction required to make a split
        reg_alpha : float
            L1 regularization term on weights
        reg_lambda : float
            L2 regularization term on weights
        n_classes : int
            Number of classes for classification
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel threads (-1 for all)
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Determine objective based on number of classes
        if n_classes == 2:
            objective = "binary:logistic"
            # Track both loss and AUROC per boosting round for plotting
            eval_metric = ["logloss", "auc"]
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        # Build XGBoost parameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": objective,
            "eval_metric": eval_metric,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 1,
        }

        # GPU support
        if use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"

        # Add num_class for multiclass
        if n_classes > 2:
            params["num_class"] = n_classes

        self.model = xgb.XGBClassifier(**params)
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = True,
    ) -> BDTClassifier:
        """Fit the BDT model.

        Parameters
        ----------
        X : np.ndarray
            Training features [N, D]
        y : np.ndarray
            Training labels [N]
        eval_set : list of tuples, optional
            Validation set(s) for early stopping
        early_stopping_rounds : int, optional
            Stop if no improvement for this many rounds
        verbose : bool
            Whether to print training progress

        Returns
        -------
        BDTClassifier
            Fitted model (self)
        """
        fit_params = {"verbose": verbose}

        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Features [N, D]

        Returns
        -------
        np.ndarray
            Predicted labels [N]
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features [N, D]

        Returns
        -------
        np.ndarray
            Class probabilities [N, n_classes]
        """
        proba = self.model.predict_proba(X)
        # For binary classification, XGBoost returns [N, 2]
        # For multiclass, it returns [N, n_classes]
        return proba

    def save(self, path: str) -> None:
        """Save model to file.

        Parameters
        ----------
        path : str
            Path to save model (will use .json format)
        """
        # Prefer sklearn API, but fall back to raw Booster on older/newer xgboost combos
        try:
            self.model.save_model(path)  # xgboost>=1.6
        except Exception:
            booster = getattr(self.model, "get_booster", None)
            if booster is None:
                raise
            booster().save_model(path)

    def load(self, path: str) -> BDTClassifier:
        """Load model from file.

        Parameters
        ----------
        path : str
            Path to load model from

        Returns
        -------
        BDTClassifier
            Loaded model (self)
        """
        # Use sklearn API (handles estimator metadata)
        self.model.load_model(path)
        self._is_fitted = True
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_

    def get_complexity_estimate(self) -> dict[str, Any]:
        """Estimate model complexity for comparison with neural networks.

        Returns
        -------
        dict
            Complexity metrics including effective parameters estimate
        """
        # Rough estimate: each split in a tree can be thought of as
        # analogous to a small number of parameters
        # A full binary tree of depth d has 2^d - 1 internal nodes
        # Each internal node has ~2 "parameters" (feature index + threshold)
        avg_nodes_per_tree = 2**self.max_depth - 1
        params_per_node = 2
        effective_params = self.n_estimators * avg_nodes_per_tree * params_per_node

        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "avg_nodes_per_tree": avg_nodes_per_tree,
            "effective_params_estimate": effective_params,
        }


def build_from_config(cfg: DictConfig, meta: Mapping[str, Any]) -> BDTClassifier:
    """Build BDT classifier from Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with classifier.model.* keys
    meta : Mapping[str, Any]
        Data metadata with n_classes key

    Returns
    -------
    BDTClassifier
        Configured BDT classifier
    """
    model_cfg = cfg.classifier.model

    # Check if GPU is available
    import torch

    use_gpu = torch.cuda.is_available() and model_cfg.get("use_gpu", False)

    return BDTClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.1),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        min_child_weight=model_cfg.get("min_child_weight", 1),
        gamma=model_cfg.get("gamma", 0.0),
        reg_alpha=model_cfg.get("reg_alpha", 0.0),
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
        n_classes=meta["n_classes"],
        random_state=cfg.classifier.trainer.get("seed", 42),
        n_jobs=model_cfg.get("n_jobs", -1),
        use_gpu=use_gpu,
    )
