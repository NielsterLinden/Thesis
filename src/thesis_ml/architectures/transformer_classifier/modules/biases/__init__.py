"""Physics-informed attention bias modules for the transformer classifier.

Public API:
    PairwiseFeatureBackbone               — shared F_ij computation
    BiasComposer, build_bias_composer     — main orchestrator
    LorentzScalarBias                     — Lorentz-scalar pairwise bias
    TypePairKinematicBias                 — learnable type-pair table
    SMInteractionBias                     — fixed SM coupling priors
    GlobalConditionedBias                 — MET-conditioned bias
    NodewiseMassBias                      — nodewise invariant mass patch
    MIAEncoder                            — more-interaction attention encoder

Shared feature utilities:
    compute_pairwise_feature_set
    VALID_FEATURES
"""

from ._features import VALID_FEATURES, compute_pairwise_feature_set
from .backbone import PairwiseFeatureBackbone
from .bias_composer import (
    BiasComposer,
    GlobalConditionedBias,
    build_bias_composer,
    parse_attention_biases,
)
from .lorentz_scalar import LorentzScalarBias
from .mia_encoder import MIAEncoder
from .nodewise_mass import NodewiseMassBias
from .type_pair_bias import SMInteractionBias, TypePairKinematicBias

__all__ = [
    "VALID_FEATURES",
    "PairwiseFeatureBackbone",
    "compute_pairwise_feature_set",
    "BiasComposer",
    "GlobalConditionedBias",
    "build_bias_composer",
    "parse_attention_biases",
    "LorentzScalarBias",
    "MIAEncoder",
    "NodewiseMassBias",
    "SMInteractionBias",
    "TypePairKinematicBias",
]
