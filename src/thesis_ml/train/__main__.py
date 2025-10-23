import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from . import DISPATCH

# Load environment variables from a local .env file (if present)
load_dotenv()


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
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
