from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from .compare_tokenizers import run_report

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs/report", config_name="compare_tokenizers", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("Starting cross-run report: compare_tokenizers")
    run_report(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
