from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import h5py
import lightning as pl
import numpy as np
from omegaconf import OmegaConf

from vw2_directact.system import VW2DirectActDataModule, VW2DirectActSystem

os.environ.setdefault("WANDB_CONSOLE", "off")


def make_temp_h5(path: Path) -> None:
    num_episodes = 4
    steps_per_episode = 12
    total = num_episodes * steps_per_episode
    episode_idx = np.repeat(np.arange(num_episodes), steps_per_episode)
    step_idx = np.tile(np.arange(steps_per_episode), num_episodes)
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("episode_idx", data=episode_idx)
        handle.create_dataset("step_idx", data=step_idx)
        handle.create_dataset("pixels", data=rng.integers(0, 255, size=(total, 32, 32, 3), dtype=np.uint8))
        handle.create_dataset("proprio", data=rng.normal(size=(total, 4)).astype(np.float32))
        handle.create_dataset("state", data=rng.normal(size=(total, 7)).astype(np.float32))
        handle.create_dataset("action", data=rng.normal(size=(total, 2)).astype(np.float32))


def make_cfg(path: Path):
    return OmegaConf.create(
        {
            "experiment_name": "smoke",
            "seed": 7,
            "output_root": str(path.parent / "outputs"),
            "logging": {"use_wandb": False, "project": "vw2_directact", "entity": None},
            "data": {
                "dataset_type": "pusht",
                "path": str(path),
                "dataset_name": "ignored",
                "cache_dir": None,
                "image_size": 32,
                "plan_horizon": 8,
                "train_split": 0.8,
                "stride": 1,
                "max_train_samples": 8,
                "max_val_samples": 4,
            },
            "model": {
                "hidden_dim": 32,
                "image_channels": 3,
                "proprio_dim": 4,
                "language_dim": 0,
                "action_dim": 2,
                "use_vq": False,
                "codebook_size": 16,
                "num_dyn_queries": 2,
                "token_chunk_horizon": 4,
                "planner_layers": 1,
                "planner_heads": 4,
                "action_decoder_layers": 1,
                "action_decoder_heads": 4,
                "action_chunk": 4,
                "freeze_encoder": False,
            },
            "train": {
                "batch_size": 2,
                "eval_batch_size": 2,
                "num_workers": 0,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "max_epochs": 1,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "grad_clip_norm": 1.0,
                "log_every_n_steps": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "save_top_k": 1,
                "resume_from": None,
                "init_from": None,
                "joint_train_encoder": False,
                "joint_train_tokenizer": False,
            },
            "loss": {
                "recon_weight": 1.0,
                "commit_weight": 0.25,
                "temporal_smooth_weight": 0.1,
                "planner_weight": 1.0,
                "action_weight": 1.0,
                "consistency_weight": 0.5,
                "video_weight": 0.0,
                "huber_delta": 1.0,
            },
            "sampling": {
                "teacher_ratio_start": 0.8,
                "teacher_ratio_end": 0.2,
                "teacher_ratio_steps": 10,
                "execute_steps": 1,
                "temperature": 1.0,
            },
            "conditioning": {"mode": "mixed"},
            "ablation": {"mode": "no_vq"},
            "eval": {
                "offline_batches": 1,
                "run_world": False,
                "num_rollouts": 0,
                "goal_offset_steps": 0,
                "max_steps": 10,
                "save_video": False,
            },
        }
    )


class SmokeTrainingTests(unittest.TestCase):
    def test_tokenizer_and_joint_fast_dev_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "synthetic_pusht.h5"
            make_temp_h5(temp_path)
            cfg = make_cfg(temp_path)

            for stage in ("tokenizer", "joint"):
                datamodule = VW2DirectActDataModule(cfg, stage)
                system = VW2DirectActSystem(cfg, stage)
                trainer = pl.Trainer(
                    accelerator="cpu",
                    devices=1,
                    precision="32-true",
                    max_epochs=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )
                trainer.fit(system, datamodule=datamodule)
                datamodule.teardown("fit")
                del trainer, system, datamodule


if __name__ == "__main__":
    unittest.main()
