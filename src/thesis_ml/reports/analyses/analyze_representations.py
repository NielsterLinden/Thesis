"""Report: Analyze Representations.

This report provides deep analysis of transformer representations for
understanding how positional encodings affect learned features.

Generates:
- Token similarity matrices per PE type
- Layer evolution PCA/t-SNE plots
- L2 norms through layers
- PE pattern visualizations

Note: This report requires loading models and running forward passes with hooks
to capture intermediate representations. It is more computationally expensive
than other reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.positional_encoding import (
    plot_pe_matrix_heatmap,
    plot_pe_per_token_patterns,
)
from thesis_ml.reports.plots.representations import (
    plot_all_similarity_matrices,
    plot_l2_norms_comparison,
    plot_layer_evolution_pca,
    plot_layer_evolution_tsne,
)
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)


class ActivationCapture:
    """Capture intermediate activations during forward pass."""

    def __init__(self):
        self.activations: dict[str, torch.Tensor] = {}
        self.hooks: list = []

    def register_hook(self, module: torch.nn.Module, name: str) -> None:
        """Register a forward hook on a module."""

        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()

        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)

    def clear(self) -> None:
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def _load_model_for_representation_analysis(
    run_dir: Path,
    data_path: Path | None = None,
    device: str | None = None,
):
    """Load model for representation analysis.

    Parameters
    ----------
    run_dir : Path
        Path to run directory
    data_path : Path | None
        Optional override for data path
    device : str | None
        Optional device string

    Returns
    -------
    tuple
        (model, cfg, device)
    """
    from thesis_ml.architectures.transformer_classifier.base import build_from_config as build_classifier
    from thesis_ml.data.h5_loader import make_classification_dataloaders
    from thesis_ml.training_loops.transformer_classifier import _gather_meta

    # Load config
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg_path.exists():
        cfg = OmegaConf.load(str(hydra_cfg_path))
    else:
        cfg_path = run_dir / "cfg.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config in {run_dir}")
        cfg = OmegaConf.load(str(cfg_path))

    # Override data path if provided
    if data_path is not None:
        cfg.data.path = str(data_path)

    # Resolve device
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Ensure meta exists
    if not hasattr(cfg, "meta") or cfg.meta is None:
        train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
        _gather_meta(cfg, meta)

    # Build model
    model = build_classifier(cfg, cfg.meta).to(dev)

    # Load weights
    best_val_path = run_dir / "best_val.pt"
    model_pt_path = run_dir / "model.pt"
    if best_val_path.exists():
        weights_path = best_val_path
    elif model_pt_path.exists():
        weights_path = model_pt_path
    else:
        raise FileNotFoundError(f"Missing weights in {run_dir}")

    checkpoint = torch.load(str(weights_path), map_location=dev, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # Handle learned PE compatibility
    pos_enc_type = cfg.classifier.model.get("positional", "unknown")
    pooling_type = cfg.classifier.model.get("pooling", "cls")
    if pos_enc_type == "learned" and pooling_type == "cls":
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("pos_enc.pe")}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model, cfg, dev


def _extract_layer_representations(
    model,
    data_loader,
    device,
    pe_type: str,
    num_batches: int = 5,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Extract layer-by-layer representations from a model.

    Parameters
    ----------
    model : nn.Module
        Loaded model
    data_loader : DataLoader
        DataLoader to get samples from
    device : torch.device
        Device to run on
    pe_type : str
        Positional encoding type
    num_batches : int
        Number of batches to process

    Returns
    -------
    tuple
        (representations dict, labels tensor)
    """
    capture = ActivationCapture()

    # Register hooks on key modules
    capture.register_hook(model.embedding, "after_embedding")

    # Only register pos_enc hook if it exists and is additive
    if model.pos_enc is not None and pe_type not in ["none", "rotary"]:
        capture.register_hook(model.pos_enc, "after_pos_enc")

    # Register hooks on each encoder block
    for block_idx, block in enumerate(model.encoder.blocks):
        capture.register_hook(block, f"after_block_{block_idx}")

    # Initialize storage
    possible_keys = ["after_embedding"]
    if model.pos_enc is not None and pe_type not in ["none", "rotary"]:
        possible_keys.append("after_pos_enc")
    possible_keys.extend([f"after_block_{i}" for i in range(len(model.encoder.blocks))])

    batch_activations: dict[str, list] = {key: [] for key in possible_keys}
    batch_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            # Unpack batch - handle different formats
            if len(batch) == 4:  # (tokens_cont, tokens_id, labels, mask)
                tokens_cont, tokens_id, labels, mask = batch
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                mask = mask.to(device)
                _ = model(tokens_cont, tokens_id, mask=mask)
            elif len(batch) == 5:  # (tokens_cont, tokens_id, globals, mask, labels)
                tokens_cont, tokens_id, globals_t, mask, labels = batch
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                mask = mask.to(device)
                _ = model(tokens_cont, tokens_id, mask=mask)
            else:  # Binned format
                if len(batch) == 3:
                    tokens, labels, mask = batch
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    _ = model(tokens, mask=mask)
                else:
                    tokens, globals_t, mask, labels = batch
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    _ = model(tokens, mask=mask)

            # Store activations
            for key in capture.activations:
                if key in batch_activations:
                    batch_activations[key].append(capture.activations[key])

            batch_labels.append(labels.cpu())
            capture.clear()

    # Concatenate batches
    representations = {}
    for key, act_list in batch_activations.items():
        if len(act_list) > 0:
            representations[key] = torch.cat(act_list, dim=0)

    all_labels = torch.cat(batch_labels, dim=0)

    # Clean up hooks
    capture.remove_hooks()

    return representations, all_labels


def _extract_positional_encoding_patterns(model, pe_type: str, device) -> dict:
    """Extract positional encoding patterns from model.

    Parameters
    ----------
    model : nn.Module
        Loaded model
    pe_type : str
        Positional encoding type
    device : torch.device
        Device

    Returns
    -------
    dict
        PE patterns (pe_matrix for additive, sin/cos for rotary)
    """
    patterns = {"pe_type": pe_type}

    if pe_type == "none":
        patterns["pe_matrix"] = None
    elif pe_type == "rotary":
        # Rotary patterns need a forward pass to populate cache
        rotary_emb = getattr(model.encoder, "rotary_emb", None)
        if rotary_emb is not None:
            patterns["rotary_emb"] = rotary_emb
            # Cache will be populated after forward pass
    else:
        # Sinusoidal or Learned
        pos_enc = model.pos_enc
        if pos_enc is not None and hasattr(pos_enc, "pe"):
            patterns["pe_matrix"] = pos_enc.pe.detach().cpu().numpy()

    return patterns


def run_report(cfg: DictConfig) -> None:
    """Analyze representations report.

    Generates deep analysis plots for transformer representations:
    - Token similarity matrices per PE type
    - Layer evolution PCA/t-SNE
    - L2 norms through layers
    - PE pattern visualizations

    Parameters
    ----------
    cfg : DictConfig
        Report configuration
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Setup report environment (load runs, create directories)
    (
        training_dir,
        inference_dir,
        training_figs_dir,
        inference_figs_dir,
        runs_df,
        per_epoch,
        order,
        report_dir,
        report_id,
        report_name,
    ) = setup_report_environment(cfg)

    logger.info(f"Loaded {len(runs_df)} runs")

    # Extract positional encoding metadata
    pos_values = []
    for run_dir in runs_df["run_dir"]:
        try:
            run_cfg, _ = _read_cfg(Path(run_dir))
            value = _extract_value_from_composed_cfg(run_cfg, "classifier.model.positional")
            pos_values.append(value)
        except Exception as e:
            logger.warning(f"Failed to extract positional from {run_dir}: {e}")
            pos_values.append(None)
    runs_df["positional"] = pos_values

    # Log unique PE types
    unique_pe = runs_df["positional"].dropna().unique()
    logger.info(f"Found positional encodings: {list(unique_pe)}")

    # Save summary CSV
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    # Determine data path
    data_path = cfg.get("data", {}).get("path", None)
    if data_path:
        data_path = Path(data_path)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ==========================================================================
    # Load models and extract representations
    # ==========================================================================

    all_representations: dict[str, dict[str, torch.Tensor]] = {}
    all_labels: dict[str, torch.Tensor] = {}
    all_pe_patterns: dict[str, dict] = {}

    # Get one representative run per PE type
    for pe_type in unique_pe:
        pe_runs = runs_df[runs_df["positional"] == pe_type]
        if len(pe_runs) == 0:
            continue

        # Use first run for this PE type
        run_dir = Path(pe_runs.iloc[0]["run_dir"])
        run_id = get_run_id(run_dir)

        logger.info(f"Loading model for {pe_type} from {run_id}...")

        try:
            model, model_cfg, dev = _load_model_for_representation_analysis(run_dir, data_path=data_path, device=str(device))

            # Create data loader for this model
            from thesis_ml.data.h5_loader import make_classification_dataloaders

            train_dl, val_dl, test_dl, meta = make_classification_dataloaders(model_cfg)

            # Extract representations
            logger.info("  Extracting layer representations...")
            representations, labels = _extract_layer_representations(
                model,
                val_dl,
                dev,
                pe_type,
                num_batches=cfg.get("analysis", {}).get("num_batches", 5),
            )

            all_representations[pe_type] = representations
            all_labels[pe_type] = labels

            # Extract PE patterns
            pe_patterns = _extract_positional_encoding_patterns(model, pe_type, dev)
            all_pe_patterns[pe_type] = pe_patterns

            logger.info(f"  Extracted {len(representations)} layers, {len(labels)} samples")

        except Exception as e:
            logger.warning(f"Failed to load model for {pe_type}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_representations:
        logger.error("No representations extracted - cannot generate plots")
        finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
        return

    # ==========================================================================
    # Generate Representation Analysis Plots
    # ==========================================================================

    # Token similarity matrices comparison
    if "similarity_matrices" in wanted:
        logger.info("Generating similarity matrices...")
        try:
            plot_all_similarity_matrices(
                all_representations,
                layer_name="after_pos_enc",  # Or auto-select
                figs_dir=inference_figs_dir,
                fig_cfg=fig_cfg,
                fname="figure-similarity_matrices_comparison",
            )
        except Exception as e:
            logger.warning(f"Error generating similarity matrices: {e}")

    # L2 norms comparison
    if "l2_norms" in wanted:
        logger.info("Generating L2 norms comparison...")
        try:
            plot_l2_norms_comparison(
                all_representations,
                figs_dir=inference_figs_dir,
                fig_cfg=fig_cfg,
                fname="figure-l2_norms_comparison",
            )
        except Exception as e:
            logger.warning(f"Error generating L2 norms: {e}")

    # Layer evolution PCA for each PE type
    if "layer_evolution_pca" in wanted:
        logger.info("Generating layer evolution PCA plots...")
        for pe_type in all_representations:
            try:
                plot_layer_evolution_pca(
                    all_representations[pe_type],
                    all_labels[pe_type],
                    pe_type,
                    figs_dir=inference_figs_dir,
                    fig_cfg=fig_cfg,
                    fname=f"figure-layer_evolution_pca_{pe_type}",
                )
            except Exception as e:
                logger.warning(f"Error generating PCA for {pe_type}: {e}")

    # Layer evolution t-SNE for each PE type
    if "layer_evolution_tsne" in wanted:
        logger.info("Generating layer evolution t-SNE plots (this may take a while)...")
        for pe_type in all_representations:
            try:
                plot_layer_evolution_tsne(
                    all_representations[pe_type],
                    all_labels[pe_type],
                    pe_type,
                    figs_dir=inference_figs_dir,
                    fig_cfg=fig_cfg,
                    fname=f"figure-layer_evolution_tsne_{pe_type}",
                )
            except Exception as e:
                logger.warning(f"Error generating t-SNE for {pe_type}: {e}")

    # ==========================================================================
    # Generate PE Pattern Plots
    # ==========================================================================

    if "pe_heatmaps" in wanted:
        logger.info("Generating PE pattern heatmaps...")
        from thesis_ml.monitoring.io_utils import save_figure

        for pe_type, patterns in all_pe_patterns.items():
            if pe_type == "none":
                continue

            if pe_type == "rotary":
                # Rotary patterns visualization would require cache access
                logger.info("  Skipping rotary patterns (requires special handling)")
                continue

            pe_matrix = patterns.get("pe_matrix")
            if pe_matrix is not None:
                try:
                    fig, ax = plot_pe_matrix_heatmap(pe_matrix, pe_type)
                    save_figure(fig, inference_figs_dir, f"figure-pe_heatmap_{pe_type}", fig_cfg)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error generating PE heatmap for {pe_type}: {e}")

    if "pe_token_patterns" in wanted:
        logger.info("Generating PE token pattern plots...")
        from thesis_ml.monitoring.io_utils import save_figure

        for pe_type, patterns in all_pe_patterns.items():
            if pe_type in ["none", "rotary"]:
                continue

            pe_matrix = patterns.get("pe_matrix")
            if pe_matrix is not None:
                try:
                    fig, ax = plot_pe_per_token_patterns(pe_matrix, pe_type)
                    save_figure(fig, inference_figs_dir, f"figure-pe_token_patterns_{pe_type}", fig_cfg)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error generating PE token patterns for {pe_type}: {e}")

    # Finalize report (manifest, backlinks, logging)
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info(f"Report complete: {report_dir}")
