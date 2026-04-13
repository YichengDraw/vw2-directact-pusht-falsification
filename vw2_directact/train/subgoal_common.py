from __future__ import annotations

import argparse
import os
import sys

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from ..subgoal_system import VW2SubgoalDataModule, VW2SubgoalSystem
from ..utils.config import ensure_stage_output_dir, load_config, save_config
from ..utils.logging import build_loggers

os.environ.setdefault("WANDB_CONSOLE", "off")
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--config-path", default=None)
    return parser


def run_subgoal_stage(stage_name: str, *, description: str) -> None:
    parser = _parser(description)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config_name, config_path=args.config_path, overrides=overrides)
    output_dir = ensure_stage_output_dir(cfg, stage_name)
    save_config(cfg, output_dir)

    system = VW2SubgoalSystem(cfg, stage_name)
    init_from = cfg.train.init_from if "init_from" in cfg.train else None
    system.load_weights_from_checkpoint(init_from)

    datamodule = VW2SubgoalDataModule(cfg)
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"{stage_name}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_weights_only=True,
        save_top_k=int(cfg.train.save_top_k),
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        max_epochs=int(cfg.train.max_epochs),
        gradient_clip_val=float(cfg.train.grad_clip_norm),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        callbacks=[checkpoint],
        logger=build_loggers(cfg, output_dir=output_dir, stage=stage_name),
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.train.resume_from)
