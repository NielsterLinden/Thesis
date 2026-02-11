#!/usr/bin/env python3
"""Evaluate VQ-VAE quality for downstream transformer use.

This script analyzes a trained VQ-VAE checkpoint to assess its quality,
including codebook utilization, reconstruction quality, and perplexity.

Usage:
    # Evaluate VQ-VAE from a run directory
    python scripts/evaluate_vq_vae.py \
        --run-dir /data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq

    # Evaluate with specific checkpoint
    python scripts/evaluate_vq_vae.py \
        --checkpoint /data/atlas/users/nterlind/checkpoints/vq_4tops_best.pt \
        --config /data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq/.hydra/config.yaml

    # With inference on test set
    python scripts/evaluate_vq_vae.py \
        --run-dir /data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq \
        --run-inference \
        --num-batches 20

    # Save detailed analysis
    python scripts/evaluate_vq_vae.py \
        --run-dir /data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq \
        --output-dir vq_analysis \
        --save-figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config_from_run(run_dir: Path):
    """Load Hydra config from run directory."""
    config_path = run_dir / ".hydra" / "config.yaml"
    cfg_fallback = run_dir / "cfg.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return OmegaConf.create(cfg)
    elif cfg_fallback.exists():
        with open(cfg_fallback) as f:
            cfg = yaml.safe_load(f)
        return OmegaConf.create(cfg)
    else:
        raise FileNotFoundError(f"No config found in {run_dir}")


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, dict]:
    """Load checkpoint and extract model state and metadata.

    Returns
    -------
    state_dict : dict
        Model state dictionary
    metadata : dict
        Additional metadata from checkpoint (if available)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}

    return state_dict, metadata


def analyze_codebook(state_dict: dict) -> dict:
    """Analyze VQ-VAE codebook from state dict.

    Parameters
    ----------
    state_dict : dict
        Model state dictionary

    Returns
    -------
    dict
        Analysis results including:
        - codebook_size: number of codebook entries
        - embedding_dim: dimension of each code
        - codebook_norm_mean: mean L2 norm of codebook entries
        - codebook_norm_std: std of L2 norm of codebook entries
        - codebook_distances: pairwise distances statistics
    """
    # Find codebook in state dict
    codebook_key = None
    for key in state_dict:
        if ("codebook" in key.lower() or "embedding" in key.lower()) and state_dict[key].ndim == 2:  # Should be [num_codes, embed_dim]
            codebook_key = key
            break

    if codebook_key is None:
        logger.warning("Could not find codebook in state dict")
        return {}

    codebook = state_dict[codebook_key]
    logger.info(f"Found codebook at '{codebook_key}': shape {codebook.shape}")

    num_codes, embed_dim = codebook.shape

    # Compute norms
    norms = torch.norm(codebook, dim=1)

    # Compute pairwise distances (sample if too large)
    if num_codes > 1000:
        # Sample for efficiency
        sample_idx = np.random.choice(num_codes, size=1000, replace=False)
        codebook_sample = codebook[sample_idx]
        distances = torch.cdist(codebook_sample, codebook_sample)
    else:
        distances = torch.cdist(codebook, codebook)

    # Remove diagonal (self-distances)
    mask = ~torch.eye(distances.shape[0], dtype=bool)
    distances = distances[mask]

    analysis = {
        "codebook_size": num_codes,
        "embedding_dim": embed_dim,
        "codebook_norm_mean": norms.mean().item(),
        "codebook_norm_std": norms.std().item(),
        "codebook_norm_min": norms.min().item(),
        "codebook_norm_max": norms.max().item(),
        "min_distance": distances.min().item(),
        "mean_distance": distances.mean().item(),
        "max_distance": distances.max().item(),
        "codebook_key": codebook_key,
    }

    return analysis


def analyze_training_metrics(run_dir: Path) -> dict:
    """Analyze training metrics from facts.

    Parameters
    ----------
    run_dir : Path
        Run directory containing facts/

    Returns
    -------
    dict
        Training analysis including:
        - best_val_loss: best validation loss
        - final_val_loss: final validation loss
        - perplexity_final: final perplexity (if available)
        - commitment_loss_final: final commitment loss (if available)
        - num_epochs: total epochs trained
    """
    scalars_path = run_dir / "facts" / "scalars.csv"

    if not scalars_path.exists():
        logger.warning(f"No scalars.csv found in {run_dir}/facts/")
        return {}

    df = pd.read_csv(scalars_path)

    # Filter validation rows
    val_df = df[df["split"] == "val"].copy()

    if len(val_df) == 0:
        logger.warning("No validation data in scalars.csv")
        return {}

    analysis = {
        "num_epochs": df["epoch"].max() + 1,
        "final_epoch": val_df["epoch"].iloc[-1],
    }

    # Extract losses
    loss_cols = [c for c in val_df.columns if "loss" in c.lower()]
    for col in loss_cols:
        if col in val_df.columns:
            analysis[f"{col}_best"] = val_df[col].min()
            analysis[f"{col}_final"] = val_df[col].iloc[-1]

    # Extract metrics (perplexity, commitment, etc.)
    metric_cols = [c for c in val_df.columns if c.startswith("metric_")]
    for col in metric_cols:
        if col in val_df.columns:
            analysis[f"{col}_final"] = val_df[col].iloc[-1]
            analysis[f"{col}_mean"] = val_df[col].mean()

    return analysis


def analyze_codebook_usage(
    model,
    data_loader,
    device: str = "cuda",
    num_batches: int | None = None,
) -> dict:
    """Analyze codebook usage by running inference.

    Parameters
    ----------
    model : nn.Module
        VQ-VAE model
    data_loader : DataLoader
        Data loader for inference
    device : str
        Device to run on
    num_batches : int, optional
        Number of batches to process (default: all)

    Returns
    -------
    dict
        Usage analysis including:
        - codebook_size: total number of codes
        - unique_codes_used: number of unique codes used
        - utilization_rate: fraction of codes used
        - code_frequencies: frequency of each code
        - perplexity: effective codebook size
    """
    model.eval()
    model.to(device)

    code_counts = {}
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            # Get batch data
            x = batch.get("tokens", batch.get("x")) if isinstance(batch, dict) else (batch[0] if isinstance(batch, list | tuple) else batch)

            x = x.to(device)

            # Forward pass to get quantized indices
            output = model(x)

            # Extract indices (depends on model structure)
            if isinstance(output, dict):
                indices = output.get("indices", output.get("quantized_indices"))
            else:
                # Try to find indices in model's latent space
                if hasattr(model, "latent_space") and hasattr(model.latent_space, "indices"):
                    indices = model.latent_space.indices
                else:
                    logger.warning("Could not extract codebook indices from model output")
                    return {}

            if indices is None:
                logger.warning("Indices are None, cannot analyze codebook usage")
                return {}

            # Count code usage
            indices_flat = indices.flatten().cpu().numpy()
            unique, counts = np.unique(indices_flat, return_counts=True)

            for code, count in zip(unique, counts, strict=False):
                code_counts[int(code)] = code_counts.get(int(code), 0) + int(count)

            total_tokens += len(indices_flat)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches...")

    # Get codebook size from model
    codebook_size = None
    if hasattr(model, "latent_space"):
        if hasattr(model.latent_space, "num_embeddings"):
            codebook_size = model.latent_space.num_embeddings
        elif hasattr(model.latent_space, "codebook_size"):
            codebook_size = model.latent_space.codebook_size

    if codebook_size is None:
        # Try to infer from max code index
        codebook_size = max(code_counts.keys()) + 1

    # Compute statistics
    unique_codes_used = len(code_counts)
    utilization_rate = unique_codes_used / codebook_size

    # Compute perplexity
    frequencies = np.array(list(code_counts.values())) / total_tokens
    perplexity = np.exp(-np.sum(frequencies * np.log(frequencies + 1e-10)))

    analysis = {
        "codebook_size": codebook_size,
        "unique_codes_used": unique_codes_used,
        "utilization_rate": utilization_rate,
        "total_tokens_processed": total_tokens,
        "perplexity": perplexity,
        "code_frequencies": code_counts,
        "unused_codes": codebook_size - unique_codes_used,
    }

    return analysis


def compute_reconstruction_quality(
    model,
    data_loader,
    device: str = "cuda",
    num_batches: int | None = None,
) -> dict:
    """Compute reconstruction quality metrics.

    Parameters
    ----------
    model : nn.Module
        VQ-VAE model
    data_loader : DataLoader
        Data loader for inference
    device : str
        Device to run on
    num_batches : int, optional
        Number of batches to process (default: all)

    Returns
    -------
    dict
        Reconstruction metrics including:
        - mse: mean squared error
        - mae: mean absolute error
        - per_feature_mse: MSE per input feature
    """
    model.eval()
    model.to(device)

    mse_sum = 0.0
    mae_sum = 0.0
    per_feature_mse = None
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            # Get batch data
            x = batch.get("tokens", batch.get("x")) if isinstance(batch, dict) else (batch[0] if isinstance(batch, list | tuple) else batch)

            x = x.to(device)
            batch_size = x.shape[0]

            # Forward pass
            output = model(x)

            # Extract reconstruction
            recon = output.get("reconstruction", output.get("recon", output.get("x_recon"))) if isinstance(output, dict) else output

            # Compute errors
            mse_batch = ((x - recon) ** 2).mean().item()
            mae_batch = (x - recon).abs().mean().item()

            mse_sum += mse_batch * batch_size
            mae_sum += mae_batch * batch_size

            # Per-feature MSE (average over batch and sequence)
            if x.ndim == 3:  # [batch, seq, features]
                per_feature_err = ((x - recon) ** 2).mean(dim=(0, 1))
            elif x.ndim == 2:  # [batch, features]
                per_feature_err = ((x - recon) ** 2).mean(dim=0)
            else:
                per_feature_err = ((x - recon) ** 2).flatten(0, -2).mean(dim=0)

            if per_feature_mse is None:
                per_feature_mse = per_feature_err.cpu().numpy() * batch_size
            else:
                per_feature_mse += per_feature_err.cpu().numpy() * batch_size

            total_samples += batch_size

    mse = mse_sum / total_samples
    mae = mae_sum / total_samples
    per_feature_mse = per_feature_mse / total_samples

    return {
        "mse": mse,
        "mae": mae,
        "rmse": np.sqrt(mse),
        "per_feature_mse": per_feature_mse.tolist(),
        "num_samples": total_samples,
    }


def plot_codebook_usage(
    code_frequencies: dict,
    codebook_size: int,
    output_path: Path | None = None,
):
    """Plot codebook usage histogram.

    Parameters
    ----------
    code_frequencies : dict
        Code ID -> frequency
    codebook_size : int
        Total codebook size
    output_path : Path, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Usage frequency histogram
    codes = np.arange(codebook_size)
    frequencies = np.array([code_frequencies.get(i, 0) for i in codes])

    axes[0].bar(codes, frequencies, width=1.0, alpha=0.7)
    axes[0].set_xlabel("Code ID")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Codebook Usage (Used: {np.sum(frequencies > 0)}/{codebook_size})")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Log-scale frequency distribution
    freq_values = np.array(list(code_frequencies.values()))
    axes[1].hist(freq_values, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Number of codes")
    axes[1].set_title("Distribution of Code Frequencies")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved codebook usage plot to {output_path}")

    plt.show()


def plot_training_curves(run_dir: Path, output_path: Path | None = None):
    """Plot training curves from scalars.csv.

    Parameters
    ----------
    run_dir : Path
        Run directory containing facts/scalars.csv
    output_path : Path, optional
        Path to save figure
    """
    scalars_path = run_dir / "facts" / "scalars.csv"

    if not scalars_path.exists():
        logger.warning(f"No scalars.csv found in {run_dir}/facts/")
        return

    df = pd.read_csv(scalars_path)

    # Separate train and val
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    # Find loss columns
    loss_cols = [c for c in df.columns if "loss" in c.lower() and c != "split"]
    metric_cols = [c for c in df.columns if c.startswith("metric_")]

    num_plots = len(loss_cols) + len(metric_cols)
    if num_plots == 0:
        logger.warning("No loss or metric columns found")
        return

    fig, axes = plt.subplots(1, min(num_plots, 3), figsize=(5 * min(num_plots, 3), 4))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot losses
    for col in loss_cols[:3]:  # Limit to 3 plots
        ax = axes[plot_idx]
        if col in train_df.columns:
            ax.plot(train_df["epoch"], train_df[col], label="train", alpha=0.7)
        if col in val_df.columns:
            ax.plot(val_df["epoch"], val_df[col], label="val", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.set_title(col.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot metrics (if room)
    for col in metric_cols:
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]
        if col in train_df.columns:
            ax.plot(train_df["epoch"], train_df[col], label="train", alpha=0.7)
        if col in val_df.columns:
            ax.plot(val_df["epoch"], val_df[col], label="val", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.set_title(col.replace("_", " ").replace("metric ", "").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training curves to {output_path}")

    plt.show()


def print_summary(
    codebook_analysis: dict,
    training_analysis: dict,
    usage_analysis: dict | None = None,
    recon_analysis: dict | None = None,
):
    """Print summary of VQ-VAE quality assessment."""
    logger.info("=" * 80)
    logger.info("VQ-VAE QUALITY ASSESSMENT")
    logger.info("=" * 80)

    # Codebook structure
    if codebook_analysis:
        logger.info("\n📊 CODEBOOK STRUCTURE:")
        logger.info(f"  Size: {codebook_analysis['codebook_size']} codes")
        logger.info(f"  Embedding dim: {codebook_analysis['embedding_dim']}")
        logger.info(f"  Norm (mean ± std): {codebook_analysis['codebook_norm_mean']:.3f} ± {codebook_analysis['codebook_norm_std']:.3f}")
        logger.info(f"  Pairwise distance: min={codebook_analysis['min_distance']:.3f}, mean={codebook_analysis['mean_distance']:.3f}, max={codebook_analysis['max_distance']:.3f}")

    # Training metrics
    if training_analysis:
        logger.info("\n📈 TRAINING METRICS:")
        logger.info(f"  Epochs trained: {training_analysis.get('num_epochs', 'N/A')}")

        # Losses
        loss_keys = [k for k in training_analysis if "loss" in k]
        for key in loss_keys:
            logger.info(f"  {key}: {training_analysis[key]:.6f}")

        # Metrics (perplexity, commitment, etc.)
        metric_keys = [k for k in training_analysis if "metric_" in k]
        for key in metric_keys:
            logger.info(f"  {key}: {training_analysis[key]:.6f}")

    # Codebook usage
    if usage_analysis:
        logger.info("\n🎯 CODEBOOK USAGE:")
        logger.info(f"  Total codes: {usage_analysis['codebook_size']}")
        logger.info(f"  Codes used: {usage_analysis['unique_codes_used']}")
        logger.info(f"  Utilization rate: {usage_analysis['utilization_rate']:.1%}")
        logger.info(f"  Unused codes: {usage_analysis['unused_codes']}")
        logger.info(f"  Perplexity: {usage_analysis['perplexity']:.2f}")
        logger.info(f"  Total tokens: {usage_analysis['total_tokens_processed']:,}")

        # Assessment
        util_rate = usage_analysis["utilization_rate"]
        if util_rate > 0.8:
            assessment = "✅ EXCELLENT - High codebook utilization"
        elif util_rate > 0.5:
            assessment = "⚠️  GOOD - Moderate codebook utilization"
        elif util_rate > 0.2:
            assessment = "⚠️  FAIR - Low codebook utilization, consider smaller codebook"
        else:
            assessment = "❌ POOR - Very low utilization, retrain with smaller codebook"

        logger.info(f"\n  Assessment: {assessment}")

    # Reconstruction quality
    if recon_analysis:
        logger.info("\n🔍 RECONSTRUCTION QUALITY:")
        logger.info(f"  MSE: {recon_analysis['mse']:.6f}")
        logger.info(f"  MAE: {recon_analysis['mae']:.6f}")
        logger.info(f"  RMSE: {recon_analysis['rmse']:.6f}")
        logger.info(f"  Samples evaluated: {recon_analysis['num_samples']:,}")

        # Feature-wise quality
        per_feat = np.array(recon_analysis["per_feature_mse"])
        logger.info(f"  Per-feature MSE: min={per_feat.min():.6f}, mean={per_feat.mean():.6f}, max={per_feat.max():.6f}")

    logger.info("\n" + "=" * 80)

    # Overall recommendation
    logger.info("\n💡 RECOMMENDATIONS:")

    recommendations = []

    if usage_analysis:
        util_rate = usage_analysis["utilization_rate"]
        if util_rate < 0.3:
            recommendations.append(f"⚠️  Low codebook utilization ({util_rate:.1%}). Consider:\n" f"   - Reducing codebook size from {usage_analysis['codebook_size']} to ~{usage_analysis['unique_codes_used']*2}\n" f"   - Increasing commitment loss weight\n" f"   - Training for more epochs")
        elif util_rate > 0.95:
            recommendations.append(f"✅ Very high utilization ({util_rate:.1%}). Codebook size may be optimal or slightly small.")

    if recon_analysis:
        mse = recon_analysis["mse"]
        if mse > 0.1:
            recommendations.append(f"⚠️  High reconstruction error (MSE={mse:.6f}). Consider:\n" f"   - Training for more epochs\n" f"   - Increasing model capacity\n" f"   - Reducing codebook size (may help with overfitting)")
        elif mse < 0.01:
            recommendations.append(f"✅ Excellent reconstruction quality (MSE={mse:.6f})")

    if training_analysis and "metric_perplex_final" in training_analysis:
        perplex = training_analysis["metric_perplex_final"]
        if usage_analysis:
            max_perplex = usage_analysis["codebook_size"]
            if perplex < max_perplex * 0.3:
                recommendations.append(f"⚠️  Low perplexity ({perplex:.2f} out of {max_perplex}). Many codes unused.")

    if not recommendations:
        recommendations.append("✅ VQ-VAE appears well-configured for downstream use.")

    for rec in recommendations:
        logger.info(rec)

    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQ-VAE quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing trained VQ-VAE",
    )
    input_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (required if using --checkpoint)",
    )

    # Analysis options
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run inference to analyze codebook usage and reconstruction",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of batches for inference (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save generated figures to output directory",
    )

    args = parser.parse_args()

    # Determine run directory and checkpoint
    if args.run_dir:
        run_dir = args.run_dir
        checkpoint_path = run_dir / "best_val.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "model.pt"
        if not checkpoint_path.exists():
            logger.error(f"No checkpoint found in {run_dir}")
            return 1

        # Load config from run
        cfg = load_config_from_run(run_dir)

    else:  # args.checkpoint
        checkpoint_path = args.checkpoint
        run_dir = None

        if not args.config:
            logger.error("--config required when using --checkpoint")
            return 1

        with open(args.config) as f:
            cfg = OmegaConf.create(yaml.safe_load(f))

    # Create output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. Analyze checkpoint directly
    # ========================================================================
    state_dict, metadata = load_checkpoint(checkpoint_path)
    codebook_analysis = analyze_codebook(state_dict)

    # ========================================================================
    # 2. Analyze training metrics (if run_dir available)
    # ========================================================================
    training_analysis = {}
    if run_dir:
        training_analysis = analyze_training_metrics(run_dir)

        # Plot training curves
        if args.save_figures and args.output_dir:
            plot_training_curves(
                run_dir,
                output_path=args.output_dir / "training_curves.png",
            )

    # ========================================================================
    # 3. Run inference (optional)
    # ========================================================================
    usage_analysis = None
    recon_analysis = None

    if args.run_inference:
        logger.info("\nRunning inference for detailed analysis...")

        try:
            # Import here to avoid import errors if not running inference
            from thesis_ml.architectures.autoencoder.base import build_from_config
            from thesis_ml.data.h5_loader import make_dataloaders

            # Load model
            model = build_from_config(cfg)
            model.load_state_dict(state_dict)
            model.to(args.device)

            # Load data
            _, val_dl, _, _ = make_dataloaders(cfg)

            # Analyze codebook usage
            logger.info("Analyzing codebook usage...")
            usage_analysis = analyze_codebook_usage(model, val_dl, device=args.device, num_batches=args.num_batches)

            # Plot codebook usage
            if usage_analysis and args.save_figures and args.output_dir:
                plot_codebook_usage(
                    usage_analysis["code_frequencies"],
                    usage_analysis["codebook_size"],
                    output_path=args.output_dir / "codebook_usage.png",
                )

            # Analyze reconstruction quality
            logger.info("Analyzing reconstruction quality...")
            recon_analysis = compute_reconstruction_quality(model, val_dl, device=args.device, num_batches=args.num_batches)

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()

    # ========================================================================
    # 4. Print summary
    # ========================================================================
    print_summary(
        codebook_analysis,
        training_analysis,
        usage_analysis,
        recon_analysis,
    )

    # ========================================================================
    # 5. Save results
    # ========================================================================
    if args.output_dir:
        results = {
            "checkpoint": str(checkpoint_path),
            "codebook_analysis": codebook_analysis,
            "training_analysis": training_analysis,
        }

        if usage_analysis:
            # Remove large frequency dict for JSON
            usage_summary = {k: v for k, v in usage_analysis.items() if k != "code_frequencies"}
            results["usage_analysis"] = usage_summary

        if recon_analysis:
            results["reconstruction_analysis"] = recon_analysis

        output_json = args.output_dir / "vq_analysis.json"
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved to {output_json}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
