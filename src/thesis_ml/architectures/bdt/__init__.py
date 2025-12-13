"""BDT (Boosted Decision Tree) classifier using XGBoost."""

from thesis_ml.architectures.bdt.base import BDTClassifier, build_from_config

__all__ = ["BDTClassifier", "build_from_config"]
