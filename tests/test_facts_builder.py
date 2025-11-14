from __future__ import annotations

import socket
import subprocess
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf

from thesis_ml.facts.builders import (
    _collect_metadata,
    _get_cuda_info,
    _get_git_commit,
    build_event_payload,
)


def test_get_git_commit_success():
    """Test git commit retrieval when in a git repo."""
    # Mock successful git commands (two calls: show-toplevel, then rev-parse)
    with mock.patch("subprocess.run") as mock_run:
        # First call: git rev-parse --show-toplevel
        # Second call: git -C <repo_root> rev-parse --short HEAD
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="/path/to/repo\n"),
            mock.Mock(returncode=0, stdout="abc1234\n"),
        ]
        result = _get_git_commit()
        assert result == "abc1234"
        assert mock_run.call_count == 2


def test_get_git_commit_not_in_repo():
    """Test git commit gracefully fails when not in a git repo."""
    with mock.patch("subprocess.run") as mock_run:
        # First call fails (not in git repo)
        mock_run.return_value = mock.Mock(returncode=128, stdout="")
        result = _get_git_commit()
        assert result is None
        # Should only call once (the show-toplevel check fails)
        assert mock_run.call_count == 1


def test_get_git_commit_timeout():
    """Test git commit handles timeout gracefully."""
    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 2)):
        result = _get_git_commit()
        assert result is None


def test_get_git_commit_file_not_found():
    """Test git commit handles missing git binary gracefully."""
    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        result = _get_git_commit()
        assert result is None


def test_get_cuda_info_available():
    """Test CUDA info when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    info = _get_cuda_info()
    assert info["available"] is True
    assert "device_count" in info
    assert "current_device" in info
    assert "device_name" in info
    assert isinstance(info["device_count"], int)
    assert info["device_count"] > 0


def test_get_cuda_info_unavailable():
    """Test CUDA info when CUDA is not available."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        info = _get_cuda_info()
        assert info["available"] is False
        assert "device_count" not in info
        assert "current_device" not in info


def test_collect_metadata_basic():
    """Test metadata collection without config."""
    with mock.patch("thesis_ml.utils.facts_builder._get_git_commit", return_value="abc123"):
        meta = _collect_metadata()

        assert "timestamp" in meta
        assert "run_id" in meta
        assert meta["git_commit"] == "abc123"
        assert meta["hostname"] == socket.gethostname()
        assert "cuda_info" in meta
        assert isinstance(meta["cuda_info"], dict)


def test_collect_metadata_with_phase1_seed():
    """Test metadata extraction from phase1.trainer.seed config."""
    cfg = OmegaConf.create(
        {
            "phase1": {
                "trainer": {
                    "seed": 42,
                },
            },
        }
    )

    meta = _collect_metadata(cfg=cfg)
    assert meta["seed"] == 42


def test_collect_metadata_with_trainer_seed():
    """Test metadata extraction from trainer.seed config."""
    cfg = OmegaConf.create(
        {
            "trainer": {
                "seed": 123,
            },
        }
    )

    meta = _collect_metadata(cfg=cfg)
    assert meta["seed"] == 123


def test_collect_metadata_with_data_seed():
    """Test metadata extraction from data.seed config."""
    cfg = OmegaConf.create(
        {
            "data": {
                "seed": 999,
            },
        }
    )

    meta = _collect_metadata(cfg=cfg)
    assert meta["seed"] == 999


def test_collect_metadata_with_hydra_job_id():
    """Test metadata extraction of Hydra job ID from environment."""
    with mock.patch.dict("os.environ", {"HYDRA_JOB_ID": "job_12345"}):
        meta = _collect_metadata()
        assert meta["hydra_job_id"] == "job_12345"


def test_build_event_payload_minimal():
    """Test payload building with minimal required args."""
    payload = build_event_payload(moment="on_start")

    assert payload["schema_version"] == 1
    assert payload["moment"] == "on_start"
    assert payload["run_dir"] == ""
    assert payload["epoch"] is None
    assert payload["step"] is None
    assert payload["split"] is None
    assert payload["train_loss"] is None
    assert payload["val_loss"] is None
    assert payload["metrics"] == {}
    assert payload["epoch_time_s"] is None
    assert payload["total_time_s"] is None
    assert payload["throughput"] is None
    assert "meta" in payload
    assert "max_memory_mib" not in payload  # Optional field


def test_build_event_payload_on_epoch_end():
    """Test payload building for on_epoch_end with typical fields."""
    payload = build_event_payload(
        moment="on_epoch_end",
        run_dir="/path/to/run",
        epoch=10,
        split="val",
        train_loss=0.5,
        val_loss=0.6,
        metrics={"perplex": 128.5, "acc": 0.95},
        epoch_time_s=12.3,
        throughput=1500.0,
    )

    assert payload["schema_version"] == 1
    assert payload["moment"] == "on_epoch_end"
    assert payload["run_dir"] == "/path/to/run"
    assert payload["epoch"] == 10
    assert payload["split"] == "val"
    assert payload["train_loss"] == 0.5
    assert payload["val_loss"] == 0.6
    assert payload["metrics"] == {"perplex": 128.5, "acc": 0.95}
    assert payload["epoch_time_s"] == 12.3
    assert payload["throughput"] == 1500.0


def test_build_event_payload_type_coercion():
    """Test that numeric values are coerced to correct types."""
    payload = build_event_payload(
        moment="on_epoch_end",
        epoch="10",  # string should be coerced to int
        train_loss="0.5",  # string should be coerced to float
        val_loss=0.6,
    )

    assert isinstance(payload["epoch"], int)
    assert payload["epoch"] == 10
    assert isinstance(payload["train_loss"], float)
    assert payload["train_loss"] == 0.5


def test_build_event_payload_with_max_memory():
    """Test optional max_memory_mib field."""
    payload = build_event_payload(
        moment="on_epoch_end",
        max_memory_mib=512.5,
    )

    assert "max_memory_mib" in payload
    assert payload["max_memory_mib"] == 512.5


def test_build_event_payload_with_histories():
    """Test history handling - both structured and flattened."""
    histories = {
        "train_loss": [1.0, 0.8, 0.5],
        "val_loss": [1.1, 0.9, 0.6],
        "perplex": [200.0, 150.0, 128.5],
    }

    payload = build_event_payload(
        moment="on_train_end",
        histories=histories,
    )

    # Check structured format (new)
    assert "histories" in payload
    assert payload["histories"]["train_loss"] == [1.0, 0.8, 0.5]
    assert payload["histories"]["val_loss"] == [1.1, 0.9, 0.6]
    assert payload["histories"]["perplex"] == [200.0, 150.0, 128.5]

    # Check flattened format (legacy compatibility)
    assert "history_train_loss" in payload
    assert payload["history_train_loss"] == [1.0, 0.8, 0.5]
    assert "history_val_loss" in payload
    assert payload["history_val_loss"] == [1.1, 0.9, 0.6]
    assert "history_perplex" in payload
    assert payload["history_perplex"] == [200.0, 150.0, 128.5]


def test_build_event_payload_histories_already_prefixed():
    """Test that history keys already prefixed with 'history_' are not double-prefixed."""
    histories = {
        "history_train_loss": [1.0, 0.5],  # Already has prefix
        "val_loss": [1.1, 0.6],  # Needs prefix
    }

    payload = build_event_payload(
        moment="on_train_end",
        histories=histories,
    )

    # Should keep existing prefix
    assert "history_train_loss" in payload
    assert payload["history_train_loss"] == [1.0, 0.5]

    # Should add prefix
    assert "history_val_loss" in payload
    assert payload["history_val_loss"] == [1.1, 0.6]


def test_build_event_payload_extra_fields():
    """Test that extra loop-specific fields are passed through."""
    payload = build_event_payload(
        moment="on_epoch_end",
        custom_metric=123.45,
        debug_info="some debug string",
        loop_specific_data=[1, 2, 3],
    )

    assert payload["custom_metric"] == 123.45
    assert payload["debug_info"] == "some debug string"
    assert payload["loop_specific_data"] == [1, 2, 3]


def test_build_event_payload_cfg_not_in_output():
    """Test that 'cfg' is used for metadata but not included in payload."""
    cfg = OmegaConf.create(
        {
            "phase1": {
                "trainer": {
                    "seed": 42,
                },
            },
        }
    )

    payload = build_event_payload(
        moment="on_start",
        cfg=cfg,
    )

    # cfg should not be in the payload
    assert "cfg" not in payload

    # But seed should be in metadata
    assert payload["meta"]["seed"] == 42


def test_build_event_payload_moment_values():
    """Test all valid moment values."""
    valid_moments = ["on_start", "on_epoch_end", "on_train_end", "on_test_end"]

    for moment in valid_moments:
        payload = build_event_payload(moment=moment)
        assert payload["moment"] == moment
        assert payload["schema_version"] == 1


def test_build_event_payload_metadata_structure():
    """Test that metadata block has expected structure."""
    with mock.patch("thesis_ml.utils.facts_builder._get_git_commit", return_value="abc123"):
        payload = build_event_payload(moment="on_start")

        meta = payload["meta"]
        assert "timestamp" in meta
        assert "run_id" in meta
        assert "git_commit" in meta
        assert meta["git_commit"] == "abc123"
        assert "hostname" in meta
        assert "cuda_info" in meta

        # Validate timestamp format (ISO 8601)
        assert "T" in meta["timestamp"]

        # Validate run_id is a UUID
        assert len(meta["run_id"]) == 36  # UUID format with hyphens
        assert meta["run_id"].count("-") == 4


def test_build_event_payload_none_vs_missing():
    """Test distinction between None values and missing fields."""
    payload = build_event_payload(
        moment="on_start",
        epoch=None,  # Explicitly None
        # train_loss not provided
    )

    # Both should result in None in the payload
    assert payload["epoch"] is None
    assert payload["train_loss"] is None

    # But optional fields like max_memory_mib should not be present
    assert "max_memory_mib" not in payload
