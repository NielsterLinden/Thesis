"""Reusable plotting functions for experiment reports.

Available modules:
- anomaly: Anomaly detection plots
- classification: Classification metrics plots (ROC, confusion matrix, etc.)
- curves: Training curve plots (loss, AUROC vs epoch)
- grids: Grid/heatmap visualizations
- positional_encoding: PE pattern visualizations
- representations: Layer representation analysis plots (PCA, t-SNE, similarity)
- scatter: Scatter plots
"""

from thesis_ml.reports.plots.classification import (
    plot_auroc_bar_by_group,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
    plot_roc_curves_grouped_by,
    plot_score_distributions,
)
from thesis_ml.reports.plots.curves import (
    plot_all_train_curves,
    plot_all_val_curves,
    plot_curves_grouped_by,
    plot_loss_vs_time,
    plot_val_auroc_curves,
    plot_val_auroc_grouped_by,
)
from thesis_ml.reports.plots.representations import (
    plot_all_similarity_matrices,
    plot_l2_norms_comparison,
    plot_layer_evolution_pca,
    plot_layer_evolution_tsne,
    plot_token_similarity_matrix,
)

__all__ = [
    # Classification plots
    "plot_auroc_bar_by_group",
    "plot_confusion_matrix",
    "plot_metrics_comparison",
    "plot_roc_curves",
    "plot_roc_curves_grouped_by",
    "plot_score_distributions",
    # Curve plots
    "plot_all_train_curves",
    "plot_all_val_curves",
    "plot_curves_grouped_by",
    "plot_loss_vs_time",
    "plot_val_auroc_curves",
    "plot_val_auroc_grouped_by",
    # Representation plots
    "plot_all_similarity_matrices",
    "plot_l2_norms_comparison",
    "plot_layer_evolution_pca",
    "plot_layer_evolution_tsne",
    "plot_token_similarity_matrix",
]
