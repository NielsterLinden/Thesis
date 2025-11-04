import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from . import DISPATCH

# Load environment variables from a local .env file (if present)
load_dotenv()


@hydra.main(config_path="../../../../../configs", config_name="config", version_base="1.3")
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
