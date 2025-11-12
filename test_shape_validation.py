"""Shape validation test for transformer classifier.

This script validates that the transformer classifier model:
1. Builds correctly from config
2. Handles forward passes for both raw and binned formats
3. Produces correct output shapes
4. Has no NaN/Inf values
5. Handles masking correctly

Usage:
    python test_shape_validation.py
"""

import torch
from omegaconf import OmegaConf

from thesis_ml.architectures.transformer_classifier.base import build_from_config


def test_raw_format():
    """Test model with raw token format."""
    print("=" * 60)
    print("Testing RAW format...")
    print("=" * 60)

    # Create minimal config
    cfg = OmegaConf.create({"classifier": {"model": {"dim": 128, "depth": 2, "heads": 4, "mlp_dim": 512, "dropout": 0.1, "positional": "sinusoidal", "norm": {"policy": "post"}, "pooling": "cls", "tokenizer": {"name": "raw", "id_embed_dim": 8}}}})

    # Create dummy meta
    meta = {"n_tokens": 18, "token_feat_dim": 4, "has_globals": False, "n_classes": 2, "num_types": 10, "vocab_size": None}  # For identity tokenizer (not used with raw)

    print("Building model...")
    model = build_from_config(cfg, meta)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model built successfully! Parameters: {param_count:,}")

    # Create dummy inputs (raw format)
    batch_size = 4
    seq_len = 18
    tokens_cont = torch.randn(batch_size, seq_len, 4)
    tokens_id = torch.randint(0, 10, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Set some padding
    mask[0, 10:] = False
    mask[1, 15:] = False

    print("\nRunning forward pass (raw format)...")
    model.eval()
    with torch.no_grad():
        logits = model(tokens_cont, tokens_id, mask=mask)

    print(f"Input shape: tokens_cont={tokens_cont.shape}, tokens_id={tokens_id.shape}, mask={mask.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: [{batch_size}, {meta['n_classes']}]")

    # Assertions
    assert logits.shape == (batch_size, meta["n_classes"]), f"Expected [{batch_size}, {meta['n_classes']}], got {logits.shape}"
    assert torch.isfinite(logits).all(), "Found NaN or Inf in logits!"
    print("\n[PASS] Shape validation passed!")
    print("[PASS] No NaN/Inf detected!")

    return model, logits


def test_binned_format():
    """Test model with binned token format."""
    print("\n" + "=" * 60)
    print("Testing BINNED format...")
    print("=" * 60)

    # Update config for binned
    cfg_binned = OmegaConf.create({"classifier": {"model": {"dim": 128, "depth": 2, "heads": 4, "mlp_dim": 512, "dropout": 0.1, "positional": "sinusoidal", "norm": {"policy": "post"}, "pooling": "cls", "tokenizer": {"name": "binned"}}}})

    meta_binned = {"n_tokens": 18, "token_feat_dim": None, "has_globals": False, "n_classes": 2, "num_types": None, "vocab_size": 886}

    print("Building model...")
    model_binned = build_from_config(cfg_binned, meta_binned)
    param_count = sum(p.numel() for p in model_binned.parameters())
    print(f"Model built successfully! Parameters: {param_count:,}")

    batch_size = 4
    seq_len = 18
    integer_tokens = torch.randint(0, 886, (batch_size, seq_len))
    mask_binned = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask_binned[0, 10:] = False

    print("\nRunning forward pass (binned format)...")
    model_binned.eval()
    with torch.no_grad():
        logits_binned = model_binned(integer_tokens, mask=mask_binned)

    print(f"Input shape: integer_tokens={integer_tokens.shape}, mask={mask_binned.shape}")
    print(f"Output shape: {logits_binned.shape}")
    assert logits_binned.shape == (batch_size, meta_binned["n_classes"]), f"Expected [{batch_size}, {meta_binned['n_classes']}], got {logits_binned.shape}"
    assert torch.isfinite(logits_binned).all(), "Found NaN or Inf in logits!"
    print("\n[PASS] Binned format validation passed!")
    print("[PASS] No NaN/Inf detected!")

    return model_binned, logits_binned


def test_mean_pooling():
    """Test model with mean pooling (no CLS token)."""
    print("\n" + "=" * 60)
    print("Testing MEAN POOLING (no CLS token)...")
    print("=" * 60)

    cfg = OmegaConf.create({"classifier": {"model": {"dim": 128, "depth": 2, "heads": 4, "mlp_dim": 512, "dropout": 0.1, "positional": "sinusoidal", "norm": {"policy": "post"}, "pooling": "mean", "tokenizer": {"name": "raw", "id_embed_dim": 8}}}})  # Mean pooling instead of CLS

    meta = {"n_tokens": 18, "token_feat_dim": 4, "has_globals": False, "n_classes": 2, "num_types": 10, "vocab_size": None}

    print("Building model with mean pooling...")
    model = build_from_config(cfg, meta)
    print("Model built successfully!")

    batch_size = 4
    seq_len = 18
    tokens_cont = torch.randn(batch_size, seq_len, 4)
    tokens_id = torch.randint(0, 10, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 10:] = False  # Some padding

    print("\nRunning forward pass with mean pooling...")
    model.eval()
    with torch.no_grad():
        logits = model(tokens_cont, tokens_id, mask=mask)

    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, meta["n_classes"]), f"Expected [{batch_size}, {meta['n_classes']}], got {logits.shape}"
    assert torch.isfinite(logits).all(), "Found NaN or Inf in logits!"
    print("\n[PASS] Mean pooling validation passed!")

    return model, logits


def main():
    """Run all shape validation tests."""
    print("\n" + "=" * 60)
    print("TRANSFORMER CLASSIFIER SHAPE VALIDATION TESTS")
    print("=" * 60)

    try:
        # Test raw format
        model_raw, logits_raw = test_raw_format()

        # Test binned format
        model_binned, logits_binned = test_binned_format()

        # Test mean pooling
        model_mean, logits_mean = test_mean_pooling()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Raw format: [PASS] (output shape {logits_raw.shape})")
        print(f"  - Binned format: [PASS] (output shape {logits_binned.shape})")
        print(f"  - Mean pooling: [PASS] (output shape {logits_mean.shape})")
        print("  - No NaN/Inf detected in any test")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
