from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from vw2_directact.system import VW2DirectActSystem
from vw2_directact.train.eval_policy import _resolve_world_image_shape
from vw2_directact.utils.rollout import prepare_policy_batch


def make_cfg(*, use_vq: bool = True, ablation_mode: str = "full"):
    return OmegaConf.create(
        {
            "experiment_name": "unit",
            "seed": 7,
            "output_root": "./vw2_directact_test_outputs",
            "logging": {"use_wandb": False, "project": "vw2_directact", "entity": None},
            "data": {
                "dataset_type": "pusht",
                "path": None,
                "dataset_name": "pusht_expert_train",
                "cache_dir": None,
                "image_size": 32,
                "plan_horizon": 8,
                "train_split": 0.9,
                "stride": 1,
                "max_train_samples": 8,
                "max_val_samples": 4,
            },
            "model": {
                "hidden_dim": 64,
                "image_channels": 3,
                "proprio_dim": 4,
                "language_dim": 0,
                "action_dim": 2,
                "use_vq": use_vq,
                "codebook_size": 32,
                "num_dyn_queries": 2,
                "token_chunk_horizon": 4,
                "planner_layers": 2,
                "planner_heads": 4,
                "action_decoder_layers": 2,
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
            "ablation": {"mode": ablation_mode},
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


class ShapeTests(unittest.TestCase):
    def test_resolve_world_image_shape_uses_dataset_pixels(self) -> None:
        class FakeDataset:
            def get_row_data(self, rows):
                self.assertEqual(rows.shape, (1,))
                return {"pixels": torch.zeros(1, 224, 224, 3, dtype=torch.uint8).numpy()}

            def assertEqual(self, left, right):
                if tuple(left) != right:
                    raise AssertionError((left, right))

        self.assertEqual(_resolve_world_image_shape(FakeDataset(), 96), (224, 224))

    def test_vq_forward_shapes(self) -> None:
        cfg = make_cfg(use_vq=True)
        system = VW2DirectActSystem(cfg, "joint")
        batch = {
            "pixels": torch.randn(2, 9, 3, 32, 32),
            "proprio": torch.randn(2, 9, 4),
            "action": torch.randn(2, 4, 2),
        }
        current = system._encode_current(batch)
        teacher = system._teacher_plan(batch, detach=True)
        self.assertEqual(tuple(current.summary.shape), (2, 64))
        self.assertEqual(tuple(teacher["plan_embeddings"].shape), (2, 4, 64))
        self.assertEqual(tuple(teacher["token_ids"].shape), (2, 4))
        actions = system.model.predict_action_chunk(
            pixels=batch["pixels"][:, 0],
            proprio=batch["proprio"][:, 0],
            mode="full",
        )
        self.assertEqual(tuple(actions.shape), (2, 4, 2))

    def test_continuous_latent_path(self) -> None:
        cfg = make_cfg(use_vq=False, ablation_mode="no_vq")
        system = VW2DirectActSystem(cfg, "joint")
        batch = {
            "pixels": torch.randn(2, 9, 3, 32, 32),
            "proprio": torch.randn(2, 9, 4),
            "action": torch.randn(2, 4, 2),
        }
        teacher = system._teacher_plan(batch, detach=True)
        self.assertEqual(int(teacher["token_ids"].numel()), 0)
        predicted = system._predicted_plan_embeddings(system._encode_current(batch))
        self.assertEqual(tuple(predicted["plan_embeddings"].shape), (2, 4, 64))

    def test_oracle_plan_override_path(self) -> None:
        cfg = make_cfg(use_vq=True)
        system = VW2DirectActSystem(cfg, "joint")
        pixels = torch.randn(2, 3, 32, 32)
        proprio = torch.randn(2, 4)
        oracle_plan = torch.randn(2, 4, 64)
        actions = system.model.predict_action_chunk(
            pixels=pixels,
            proprio=proprio,
            mode="oracle",
            plan_override=oracle_plan,
        )
        self.assertEqual(tuple(actions.shape), (2, 4, 2))
        with self.assertRaises(ValueError):
            system.model.predict_action_chunk(
                pixels=pixels,
                proprio=proprio,
                mode="oracle",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the rollout device regression test")
    def test_prepare_policy_batch_keeps_pixels_on_target_device(self) -> None:
        device = torch.device("cuda")
        info_dict = {
            "pixels": torch.randint(0, 255, (2, 1, 32, 32, 3), dtype=torch.uint8),
            "proprio": torch.randn(2, 1, 4),
        }
        batch = prepare_policy_batch(info_dict, image_size=32, device=device)
        self.assertIsNotNone(batch["pixels"])
        self.assertEqual(batch["pixels"].device.type, "cuda")
        self.assertEqual(batch["proprio"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
