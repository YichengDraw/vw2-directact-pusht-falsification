from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def load_config(config_name: str, *, config_path: str | None = None, overrides: Sequence[str] = ()) -> DictConfig:
    path = Path(config_path) if config_path else (_config_dir() / f"{config_name}.yaml")
    cfg = OmegaConf.load(path)
    cli_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, cli_cfg)


def ensure_stage_output_dir(cfg: DictConfig, stage: str) -> Path:
    root = Path(cfg.output_root)
    directory = root / cfg.experiment_name / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_config(cfg: DictConfig, output_dir: Path) -> None:
    OmegaConf.save(cfg, output_dir / "config.yaml")
