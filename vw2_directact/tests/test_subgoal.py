from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import h5py
import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vw2_directact.data import PushTSubgoalDataset
from vw2_directact.subgoal_system import VW2SubgoalDataModule, VW2SubgoalSystem
from vw2_directact.train.eval_policy import (
    _merge_batch_max_steps,
    _require_requested_rollouts,
    _resolve_world_dataset as _resolve_directact_world_dataset,
)
from vw2_directact.train.eval_subgoal_policy import (
    _resolve_world_dataset as _resolve_subgoal_world_dataset,
    _select_eval_starts as _select_subgoal_eval_starts,
    _subgoal_offline_metrics,
    _teacher_gate,
)
from vw2_directact.utils.rollout import DirectActPolicy, SubgoalPolicy

os.environ.setdefault("WANDB_CONSOLE", "off")


def make_temp_h5(path: Path) -> None:
    num_episodes = 4
    steps_per_episode = 16
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
            "experiment_name": "subgoal_smoke",
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
            "subgoal": {
                "history_steps": 4,
                "subgoal_dim": 32,
                "history_layers": 1,
                "history_heads": 4,
                "future_layers": 1,
                "future_heads": 4,
                "max_horizon": 16,
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
            },
            "loss": {
                "action_weight": 1.0,
                "huber_delta": 1.0,
                "subgoal_weight": 1.0,
                "actdistill_weight": 1.0,
                "var_weight": 1.0,
                "nce_weight": 0.5,
                "subgoal_variance_floor": 1.0,
                "nce_temperature": 0.1,
            },
            "sampling": {
                "teacher_ratio_start": 0.8,
                "teacher_ratio_end": 0.2,
                "teacher_ratio_steps": 10,
                "execute_steps": 1,
                "temperature": 1.0,
            },
            "conditioning": {"mode": "mixed"},
            "ablation": {"mode": "full"},
            "eval": {
                "offline_batches": 1,
                "run_world": False,
                "num_rollouts": 0,
                "goal_offset_steps": 8,
                "max_steps": 10,
                "save_video": False,
                "save_video_count": 0,
                "rollout_batch_size": 2,
                "subgoal_execute_actions_per_plan_sweep": [1, 2],
            },
        }
    )


class DummyDirectActModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def predict_action_chunk(self, *, plan_override: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if plan_override is None:
            raise ValueError("plan_override is required for this test.")
        values = plan_override[:, 0, 0].to(self.anchor.device, dtype=torch.float32)
        return torch.stack([values, values + 0.25], dim=1).unsqueeze(-1)


class DummySubgoalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def predict_action_chunk(self, *, oracle_subgoal: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if oracle_subgoal is None:
            raise ValueError("oracle_subgoal is required for this test.")
        values = oracle_subgoal[:, 0].to(self.anchor.device, dtype=torch.float32)
        return torch.stack([values, values + 0.25], dim=1).unsqueeze(-1)


class FakeWorldDataset:
    column_names = ["episode_idx", "step_idx"]

    def __init__(self, episode_ids: np.ndarray, step_idx: np.ndarray) -> None:
        self._episode_ids = episode_ids
        self._step_idx = step_idx

    def get_col_data(self, name: str) -> np.ndarray:
        if name == "episode_idx":
            return self._episode_ids
        if name == "step_idx":
            return self._step_idx
        raise KeyError(name)

    def get_row_data(self, rows: list[int]) -> dict[str, np.ndarray]:
        index = np.asarray(rows, dtype=np.int64)
        return {"episode_idx": self._episode_ids[index], "step_idx": self._step_idx[index]}


class SubgoalTests(unittest.TestCase):
    def test_directact_policy_uses_stepwise_oracle_plan(self) -> None:
        policy = DirectActPolicy(
            model=DummyDirectActModel(),
            image_size=4,
            execute_steps=1,
            mode="oracle",
            oracle_plan_embeddings_by_step=torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]], dtype=torch.float32),
        )
        info = {"pixels": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)}
        values = [float(policy.get_action(info)[0, 0]) for _ in range(3)]
        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_directact_policy_replans_oracle_plan_after_execute_window(self) -> None:
        policy = DirectActPolicy(
            model=DummyDirectActModel(),
            image_size=4,
            execute_steps=2,
            mode="oracle",
            oracle_plan_embeddings_by_step=torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]], dtype=torch.float32),
        )
        info = {"pixels": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)}
        values = [float(policy.get_action(info)[0, 0]) for _ in range(4)]
        self.assertEqual(values, [1.0, 1.25, 3.0, 3.25])

    def test_subgoal_policy_uses_stepwise_oracle_subgoal(self) -> None:
        policy = SubgoalPolicy(
            model=DummySubgoalModel(),
            image_size=4,
            history_steps=4,
            action_dim=1,
            execute_steps=1,
            horizon_steps=8,
            mode="oracle",
            oracle_subgoals_by_step=torch.tensor([[[5.0], [6.0], [7.0]]], dtype=torch.float32),
            bootstrap_history_pixels=torch.zeros(1, 4, 3, 4, 4, dtype=torch.float32),
            bootstrap_history_proprio=None,
            bootstrap_prev_actions=torch.zeros(1, 4, 1, dtype=torch.float32),
        )
        info = {"pixels": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)}
        values = [float(policy.get_action(info)[0, 0]) for _ in range(3)]
        self.assertEqual(values, [5.0, 6.0, 7.0])

    def test_subgoal_policy_replans_oracle_subgoal_after_execute_window(self) -> None:
        policy = SubgoalPolicy(
            model=DummySubgoalModel(),
            image_size=4,
            history_steps=4,
            action_dim=1,
            execute_steps=2,
            horizon_steps=8,
            mode="oracle",
            oracle_subgoals_by_step=torch.tensor([[[5.0], [6.0], [7.0], [8.0]]], dtype=torch.float32),
            bootstrap_history_pixels=torch.zeros(1, 4, 3, 4, 4, dtype=torch.float32),
            bootstrap_history_proprio=None,
            bootstrap_prev_actions=torch.zeros(1, 4, 1, dtype=torch.float32),
        )
        info = {"pixels": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)}
        values = [float(policy.get_action(info)[0, 0]) for _ in range(4)]
        self.assertEqual(values, [5.0, 5.25, 7.0, 7.25])

    def test_subgoal_eval_start_accepts_exact_terminal_window(self) -> None:
        cfg = make_cfg(Path("unused.h5"))
        cfg.eval.num_rollouts = 1
        cfg.eval.max_steps = 100
        cfg.eval.goal_offset_steps = 8
        cfg.data.plan_horizon = 8
        cfg.subgoal.history_steps = 4
        episode_ids = np.zeros(111, dtype=np.int64)
        step_idx = np.arange(111, dtype=np.int64)
        dataset = FakeWorldDataset(episode_ids, step_idx)

        episodes, starts = _select_subgoal_eval_starts(dataset, cfg)

        self.assertEqual(episodes.tolist(), [0])
        self.assertEqual(starts.tolist(), [3])

    def test_world_eval_fails_when_valid_starts_are_fewer_than_requested(self) -> None:
        cfg = make_cfg(Path("unused.h5"))
        cfg.eval.num_rollouts = 3

        with self.assertRaisesRegex(ValueError, "requested 3 rollouts but only 2 valid starts"):
            _require_requested_rollouts(np.asarray([0, 1]), cfg, context="Push-T test evaluation")

    def test_world_eval_requires_positive_rollout_count(self) -> None:
        cfg = make_cfg(Path("unused.h5"))
        cfg.eval.num_rollouts = 0

        with self.assertRaisesRegex(ValueError, "requires eval.num_rollouts to be positive"):
            _require_requested_rollouts(np.asarray([], dtype=np.int64), cfg, context="Push-T test evaluation")

    def test_world_eval_rejects_inconsistent_batch_max_steps(self) -> None:
        current = _merge_batch_max_steps(None, {"max_steps": 100}, context="Push-T test evaluation")
        self.assertEqual(current, 100)

        with self.assertRaisesRegex(ValueError, "inconsistent max_steps"):
            _merge_batch_max_steps(current, {"max_steps": 99}, context="Push-T test evaluation")

    def test_teacher_gate_requires_world_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing world metrics"):
            _teacher_gate({"world": {"1": {}, "2": {}}})

    def test_subgoal_dataset_and_predict_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "synthetic_pusht.h5"
            make_temp_h5(temp_path)
            cfg = make_cfg(temp_path)
            dataset = PushTSubgoalDataset(
                path=str(temp_path),
                image_size=32,
                history_steps=4,
                future_horizon=8,
                action_horizon=4,
                train=True,
                train_split=0.8,
                seed=7,
                stride=1,
                max_samples=4,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["history_pixels"].shape), (4, 3, 32, 32))
            self.assertEqual(tuple(sample["future_pixels"].shape), (8, 3, 32, 32))
            self.assertEqual(tuple(sample["prev_actions"].shape), (4, 2))
            self.assertEqual(tuple(sample["action"].shape), (4, 2))

            system = VW2SubgoalSystem(cfg, "joint_subgoal")
            actions = system.model.predict_action_chunk(
                history_pixels=sample["history_pixels"].unsqueeze(0),
                history_proprio=sample["history_proprio"].unsqueeze(0),
                prev_actions=sample["prev_actions"].unsqueeze(0),
                horizon_steps=torch.tensor([8]),
                mode="student",
            )
            self.assertEqual(tuple(actions.shape), (1, 4, 2))

    def test_subgoal_training_and_offline_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "synthetic_pusht.h5"
            make_temp_h5(temp_path)
            cfg = make_cfg(temp_path)

            for stage in ("teacher_oracle", "student_predictor", "joint_subgoal"):
                datamodule = VW2SubgoalDataModule(cfg)
                system = VW2SubgoalSystem(cfg, stage)
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

            joint_system = VW2SubgoalSystem(cfg, "joint_subgoal")
            metrics = _subgoal_offline_metrics(joint_system, cfg, mode="student")
            self.assertIn("action_mse", metrics)
            self.assertIn("retrieval_top1", metrics)
            self.assertIn("retrieval_top1_shuffled", metrics)

    def test_strict_checkpoint_load_rejects_incompatible_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "synthetic_pusht.h5"
            make_temp_h5(temp_path)
            cfg = make_cfg(temp_path)
            checkpoint_path = Path(temp_dir) / "bad.ckpt"
            torch.save({"state_dict": {"not_a_model_key": torch.zeros(1)}}, checkpoint_path)

            system = VW2SubgoalSystem(cfg, "joint_subgoal")
            with self.assertRaisesRegex(RuntimeError, "did not match"):
                system.load_weights_from_checkpoint(str(checkpoint_path), strict=True)

    def test_world_dataset_resolution_fails_when_h5_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing_pusht.h5"
            cfg = make_cfg(missing_path)

            with self.assertRaisesRegex(FileNotFoundError, "Push-T HDF5 dataset not found"):
                _resolve_directact_world_dataset(cfg)
            with self.assertRaisesRegex(FileNotFoundError, "Push-T HDF5 dataset not found"):
                _resolve_subgoal_world_dataset(cfg)


if __name__ == "__main__":
    unittest.main()
