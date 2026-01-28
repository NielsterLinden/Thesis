from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from . import DISPATCH

# Load environment variables from a local .env file (if present)
load_dotenv()

# Register custom resolver for zero-padded job numbers (e.g., job00, job01)
# This avoids sorting issues when training more than 10 models
OmegaConf.register_new_resolver("zpad", lambda x, width=2: str(x).zfill(width), replace=True)

# Calculate absolute path to configs directory (repo root is 5 levels up from this file)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


@hydra.main(config_path=str(CONFIGS_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Guardrail: catch legacy config keys from pre-refactor code
    bad = []
    if "phase1" in cfg:
        if "model" in cfg.phase1:
            bad.append("cfg.phase1.model (no longer used; params moved to encoder/decoder/latent_space)")
        if "tokenizer" in cfg.phase1:
            bad.append("cfg.phase1.tokenizer (rename to cfg.phase1.latent_space)")
        if "full_model" in cfg.phase1:
            bad.append("cfg.phase1.full_model (no longer used; params moved to encoder/decoder/latent_space)")
        if "ae" in cfg.phase1:
            bad.append("cfg.phase1.ae (no longer used; latent_dim moved to latent_space configs)")
    if bad:
        raise ValueError("Legacy config keys detected:\n- " + "\n- ".join(bad))

    # Use only top-level cfg.loop
    loop = cfg.get("loop")
    if not loop:
        raise ValueError(f"Missing 'loop' in config. Available: {list(DISPATCH)}")
    fn = DISPATCH.get(loop)
    if fn is None:
        raise ValueError(f"Unknown loop='{loop}'. Options: {list(DISPATCH)}")
    return fn(cfg)


if __name__ == "__main__":
    main()
