from __future__ import annotations

from pathlib import Path

from lightning.pytorch.loggers import CSVLogger


def build_loggers(cfg, *, output_dir: Path, stage: str):
    loggers = [CSVLogger(save_dir=str(output_dir), name="csv")]
    if getattr(cfg.logging, "use_wandb", False):
        try:
            from lightning.pytorch.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=cfg.logging.project,
                    entity=cfg.logging.entity,
                    name=f"{cfg.experiment_name}-{stage}",
                    save_dir=str(output_dir),
                )
            )
        except Exception:
            pass
    return loggers
