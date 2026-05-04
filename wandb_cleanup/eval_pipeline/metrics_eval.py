"""Re-export shim — canonical implementation at thesis_ml.utils.eval_metrics.

stage_b_inference.py and any other callers in wandb_cleanup/ can keep their
existing `from metrics_eval import ...` imports unchanged.
"""

from thesis_ml.utils.eval_metrics import (  # noqa: F401
    _binary_scores_probs,
    brier_binary,
    confusion_dict,
    cross_entropy_mean,
    ece_multiclass,
    eps_s_at_background_rejection,
    flatten_metrics,
    flops_analytic_transformer,
    log_loss_mc,
    partial_auroc_fpr,
    partial_auroc_tpr,
    per_class_auroc,
    score_histograms_binary,
    tier3_placeholder,
)
