from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .data import CalvinSequenceDataset, PushTSequenceDataset, VideoPretrainDataset
from .data.common import build_torch_split
from .models import (
    ActionDecoder,
    ForwardConsistencyModel,
    FutureFeatureHead,
    ObservationEncoder,
    PlannerTransformer,
    TemporalDynamicsTokenizer,
)
from .models.losses import action_huber_loss, token_accuracy


@dataclass
class EncodedBatch:
    summary: torch.Tensor
    proprio_token: torch.Tensor | None
    language_token: torch.Tensor | None


class VW2DirectActDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, stage_name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage_name = stage_name
        self.train_dataset = None
        self.val_dataset = None

    def _make_dataset(self, *, train: bool):
        max_samples = self.cfg.data.max_train_samples if train else self.cfg.data.max_val_samples
        shared = dict(
            image_size=int(self.cfg.data.image_size),
            plan_horizon=int(self.cfg.data.plan_horizon),
            action_horizon=int(self.cfg.model.action_chunk),
            train=train,
            train_split=float(self.cfg.data.train_split),
            seed=int(self.cfg.seed),
            stride=int(self.cfg.data.stride),
            max_samples=None if max_samples is None else int(max_samples),
        )
        dataset_type = str(self.cfg.data.dataset_type)
        if self.stage_name == "tokenizer":
            if dataset_type == "pusht":
                return VideoPretrainDataset(
                    dataset_type=dataset_type,
                    path=self.cfg.data.path,
                    dataset_name=self.cfg.data.dataset_name,
                    cache_dir=self.cfg.data.cache_dir,
                    **shared,
                )
            return VideoPretrainDataset(
                dataset_type=dataset_type,
                path=self.cfg.data.path,
                **shared,
            )
        if dataset_type == "pusht":
            return PushTSequenceDataset(
                path=self.cfg.data.path,
                dataset_name=self.cfg.data.dataset_name,
                cache_dir=self.cfg.data.cache_dir,
                **shared,
            )
        if dataset_type == "calvin":
            return CalvinSequenceDataset(path=self.cfg.data.path, **shared)
        raise ValueError(f"Unsupported dataset_type={dataset_type!r}.")

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = self._make_dataset(train=True)
        if self.val_dataset is None:
            self.val_dataset = self._make_dataset(train=False)

    def train_dataloader(self):
        return build_torch_split(
            self.train_dataset,
            batch_size=int(self.cfg.train.batch_size),
            num_workers=int(self.cfg.train.num_workers),
            train=True,
        )

    def val_dataloader(self):
        return build_torch_split(
            self.val_dataset,
            batch_size=int(self.cfg.train.eval_batch_size),
            num_workers=int(self.cfg.train.num_workers),
            train=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        for dataset in (self.train_dataset, self.val_dataset):
            close = getattr(dataset, "close", None)
            if close is not None:
                close()


class VW2DirectActModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        plan_tokens = int(cfg.data.plan_horizon // cfg.model.token_chunk_horizon) * int(cfg.model.num_dyn_queries)
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.plan_horizon = int(cfg.data.plan_horizon)
        self.action_chunk = int(cfg.model.action_chunk)
        self.use_vq = bool(cfg.model.use_vq)
        self.obs_encoder = ObservationEncoder(
            hidden_dim=int(cfg.model.hidden_dim),
            image_channels=int(cfg.model.image_channels),
            proprio_dim=int(cfg.model.proprio_dim),
            language_dim=int(cfg.model.language_dim),
            freeze_encoder=bool(cfg.model.freeze_encoder),
        )
        self.tokenizer = TemporalDynamicsTokenizer(
            hidden_dim=int(cfg.model.hidden_dim),
            chunk_horizon=int(cfg.model.token_chunk_horizon),
            num_queries=int(cfg.model.num_dyn_queries),
            use_vq=bool(cfg.model.use_vq),
            codebook_size=int(cfg.model.codebook_size),
            heads=int(cfg.model.planner_heads),
        )
        self.planner = PlannerTransformer(
            hidden_dim=int(cfg.model.hidden_dim),
            plan_tokens=plan_tokens,
            use_vq=bool(cfg.model.use_vq),
            codebook_size=int(cfg.model.codebook_size),
            layers=int(cfg.model.planner_layers),
            heads=int(cfg.model.planner_heads),
        )
        self.action_decoder = ActionDecoder(
            hidden_dim=int(cfg.model.hidden_dim),
            action_dim=int(cfg.model.action_dim),
            action_chunk=int(cfg.model.action_chunk),
            layers=int(cfg.model.action_decoder_layers),
            heads=int(cfg.model.action_decoder_heads),
        )
        self.consistency_model = ForwardConsistencyModel(
            hidden_dim=int(cfg.model.hidden_dim),
            action_dim=int(cfg.model.action_dim),
            action_chunk=int(cfg.model.action_chunk),
        )
        self.video_head = FutureFeatureHead(
            hidden_dim=int(cfg.model.hidden_dim),
            plan_horizon=int(cfg.data.plan_horizon),
        )

    def predict_action_chunk(
        self,
        *,
        pixels: torch.Tensor,
        proprio: torch.Tensor | None = None,
        gripper_pixels: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
        temperature: float = 1.0,
        mode: str = "predfuture",
        plan_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = self.obs_encoder.encode_observation(
            pixels=pixels,
            gripper_pixels=gripper_pixels,
            proprio=proprio,
            language=language,
        )
        conditioning_mode = mode if mode in {"bc", "oracle", "predfuture"} else "predfuture"
        if conditioning_mode == "bc":
            plan_embeddings = pixels.new_zeros((pixels.shape[0], 0, self.hidden_dim))
        elif conditioning_mode == "oracle":
            if plan_override is None:
                raise ValueError("Oracle conditioning requires plan_override.")
            plan_embeddings = plan_override
        elif plan_override is not None:
            plan_embeddings = plan_override
        else:
            generated = self.planner.generate(current["summary"], temperature=temperature)
            if self.use_vq:
                plan_embeddings = self.tokenizer.lookup(generated["token_ids"])
            else:
                plan_embeddings = generated["latents"]
        return self.action_decoder(
            obs_summary=current["summary"],
            plan_tokens=plan_embeddings,
            proprio_token=current["proprio_token"],
            language_token=current["language_token"],
        )


class VW2DirectActSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig, stage_name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage_name = stage_name
        self.model = VW2DirectActModel(cfg)
        self.plan_chunks = int(cfg.data.plan_horizon // cfg.model.token_chunk_horizon)
        self.plan_tokens = self.plan_chunks * int(cfg.model.num_dyn_queries)
        self.ablation_mode = str(cfg.ablation.mode)
        self.conditioning_mode = self._resolve_conditioning_mode()
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True), "stage_name": stage_name})
        self._apply_freeze_policy()

    def _resolve_conditioning_mode(self) -> str:
        conditioning = getattr(self.cfg, "conditioning", None)
        if conditioning is not None and "mode" in conditioning:
            return str(conditioning.mode)
        if self.ablation_mode == "bc":
            return "bc"
        if self.stage_name == "action":
            return "oracle"
        return "mixed"

    def _apply_freeze_policy(self) -> None:
        modules = {
            "encoder": self.model.obs_encoder,
            "tokenizer": self.model.tokenizer,
            "planner": self.model.planner,
            "action": self.model.action_decoder,
            "consistency": self.model.consistency_model,
            "video": self.model.video_head,
        }
        for module in modules.values():
            module.requires_grad_(False)

        if self.stage_name == "tokenizer":
            self.model.obs_encoder.requires_grad_(True)
            self.model.tokenizer.requires_grad_(True)
            return

        if self.stage_name == "planner":
            self.model.planner.requires_grad_(True)
            if not bool(self.cfg.model.freeze_encoder):
                self.model.obs_encoder.requires_grad_(True)
            return

        if self.stage_name == "action":
            self.model.action_decoder.requires_grad_(True)
            self.model.consistency_model.requires_grad_(True)
            self.model.video_head.requires_grad_(True)
            if not bool(self.cfg.model.freeze_encoder):
                self.model.obs_encoder.requires_grad_(True)
            return

        self.model.planner.requires_grad_(True)
        self.model.action_decoder.requires_grad_(True)
        self.model.consistency_model.requires_grad_(True)
        self.model.video_head.requires_grad_(True)
        if bool(self.cfg.train.joint_train_encoder):
            self.model.obs_encoder.requires_grad_(True)
        if bool(self.cfg.train.joint_train_tokenizer):
            self.model.tokenizer.requires_grad_(True)

    def load_weights_from_checkpoint(self, checkpoint_path: str | None, *, strict: bool = False) -> None:
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            message = (
                f"Checkpoint {checkpoint_path} did not match the current model. "
                f"Missing keys: {incompatible.missing_keys}. "
                f"Unexpected keys: {incompatible.unexpected_keys}."
            )
            if strict:
                raise RuntimeError(message)
            print(f"Warning: {message}")

    def _current_step(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value[:, 0]

    def _future_steps(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value[:, 1 : 1 + int(self.cfg.data.plan_horizon)]

    def _encode_current(self, batch: dict[str, torch.Tensor]) -> EncodedBatch:
        encoded = self.model.obs_encoder.encode_observation(
            pixels=self._current_step(batch.get("pixels")),
            gripper_pixels=self._current_step(batch.get("gripper_pixels")),
            proprio=self._current_step(batch.get("proprio")),
            language=self._current_step(batch.get("language")),
        )
        return EncodedBatch(
            summary=encoded["summary"],
            proprio_token=encoded["proprio_token"],
            language_token=encoded["language_token"],
        )

    def _encode_future(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.model.obs_encoder.encode_sequence(
            pixels=self._future_steps(batch.get("pixels")),
            gripper_pixels=self._future_steps(batch.get("gripper_pixels")),
            proprio=self._future_steps(batch.get("proprio")),
            language=self._future_steps(batch.get("language")),
        )
        return encoded["summary_sequence"]

    def _teacher_plan(self, batch: dict[str, torch.Tensor], detach: bool = True) -> dict[str, torch.Tensor]:
        context = torch.no_grad if detach else nullcontext
        with context():
            future_summary = self._encode_future(batch)
            chunks = future_summary.reshape(
                future_summary.shape[0],
                self.plan_chunks,
                int(self.cfg.model.token_chunk_horizon),
                future_summary.shape[-1],
            )
            tokenizer_output = self.model.tokenizer(chunks)
        if detach:
            tokenizer_output["plan_embeddings"] = tokenizer_output["plan_embeddings"].detach()
            if tokenizer_output["token_ids"].numel() > 0:
                tokenizer_output["token_ids"] = tokenizer_output["token_ids"].detach()
            future_summary = future_summary.detach()
        tokenizer_output["future_summary"] = future_summary
        return tokenizer_output

    def _teacher_ratio(self) -> float:
        start = float(self.cfg.sampling.teacher_ratio_start)
        end = float(self.cfg.sampling.teacher_ratio_end)
        total_steps = max(int(self.cfg.sampling.teacher_ratio_steps), 1)
        progress = min(float(self.global_step) / total_steps, 1.0)
        return start + (end - start) * progress

    def _plan_loss(self, current: EncodedBatch, teacher_plan: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.cfg.model.use_vq):
            prediction = self.model.planner.forward_train(current.summary, teacher_plan["plan_embeddings"])
            return F.mse_loss(prediction, teacher_plan["plan_embeddings"]), torch.zeros((), device=self.device)

        logits = self.model.planner.forward_train(current.summary, teacher_plan["token_ids"])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), teacher_plan["token_ids"].reshape(-1))
        acc = token_accuracy(logits, teacher_plan["token_ids"])
        return loss, acc

    def _predicted_plan_embeddings(self, current: EncodedBatch) -> dict[str, torch.Tensor]:
        generated = self.model.planner.generate(
            current.summary,
            temperature=float(self.cfg.sampling.temperature),
        )
        if bool(self.cfg.model.use_vq):
            generated["plan_embeddings"] = self.model.tokenizer.lookup(generated["token_ids"])
        else:
            generated["plan_embeddings"] = generated["latents"]
        return generated

    def _mixed_plan(self, teacher_embeddings: torch.Tensor, predicted_embeddings: torch.Tensor) -> torch.Tensor:
        ratio = self._teacher_ratio()
        mask = (torch.rand(teacher_embeddings.shape[:2], device=teacher_embeddings.device) < ratio).unsqueeze(-1)
        return torch.where(mask, teacher_embeddings, predicted_embeddings)

    def _plan_for_conditioning(
        self,
        current: EncodedBatch,
        teacher_plan: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conditioning_mode == "bc":
            return (
                self._empty_plan(batch_size),
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
            )

        if self.stage_name == "action":
            if self.conditioning_mode == "predfuture":
                predicted_plan = self._predicted_plan_embeddings(current)
                return (
                    predicted_plan["plan_embeddings"],
                    torch.zeros((), device=self.device),
                    torch.zeros((), device=self.device),
                )
            return (
                teacher_plan["plan_embeddings"],
                torch.zeros((), device=self.device),
                torch.zeros((), device=self.device),
            )

        plan_loss, plan_acc = self._plan_loss(current, teacher_plan)
        if self.conditioning_mode == "oracle":
            plan_embeddings = teacher_plan["plan_embeddings"]
        elif self.conditioning_mode == "predfuture":
            predicted_plan = self._predicted_plan_embeddings(current)
            plan_embeddings = predicted_plan["plan_embeddings"]
        else:
            predicted_plan = self._predicted_plan_embeddings(current)
            plan_embeddings = self._mixed_plan(teacher_plan["plan_embeddings"], predicted_plan["plan_embeddings"])
        return plan_embeddings, plan_loss, plan_acc

    def _action_and_aux_losses(
        self,
        current: EncodedBatch,
        *,
        plan_embeddings: torch.Tensor,
        teacher_plan: dict[str, torch.Tensor],
        target_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted_actions = self.model.action_decoder(
            obs_summary=current.summary,
            plan_tokens=plan_embeddings,
            proprio_token=current.proprio_token,
            language_token=current.language_token,
        )
        target_future = teacher_plan["future_summary"][:, int(self.cfg.model.action_chunk) - 1]
        losses = {
            "action_loss": action_huber_loss(
                predicted_actions,
                target_actions,
                delta=float(self.cfg.loss.huber_delta),
            ),
            "action_mse": F.mse_loss(predicted_actions, target_actions),
        }
        if float(self.cfg.loss.consistency_weight) > 0 and self.ablation_mode != "no_consistency":
            consistency_prediction = self.model.consistency_model(current.summary, predicted_actions)
            losses["consistency_loss"] = F.mse_loss(consistency_prediction, target_future)
        else:
            losses["consistency_loss"] = torch.zeros((), device=self.device)

        if float(self.cfg.loss.video_weight) > 0 and plan_embeddings.numel() > 0:
            video_prediction = self.model.video_head(current.summary, plan_embeddings)
            losses["video_loss"] = F.l1_loss(video_prediction, teacher_plan["future_summary"])
        else:
            losses["video_loss"] = torch.zeros((), device=self.device)
        losses["predicted_actions"] = predicted_actions
        return losses

    def _empty_plan(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, 0, int(self.cfg.model.hidden_dim), device=self.device)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="val")

    def _shared_step(self, batch: dict[str, torch.Tensor], *, prefix: str) -> torch.Tensor:
        current = self._encode_current(batch)
        target_actions = batch["action"][:, : int(self.cfg.model.action_chunk)]

        if self.stage_name == "tokenizer":
            teacher_plan = self._teacher_plan(batch, detach=False)
            loss = (
                float(self.cfg.loss.recon_weight) * teacher_plan["recon_loss"]
                + float(self.cfg.loss.commit_weight) * teacher_plan["commitment_loss"]
                + float(self.cfg.loss.temporal_smooth_weight) * teacher_plan["temporal_smooth_loss"]
            )
            self.log_dict(
                {
                    f"{prefix}/loss": loss,
                    f"{prefix}_loss": loss,
                    f"{prefix}/recon_loss": teacher_plan["recon_loss"],
                    f"{prefix}/commitment_loss": teacher_plan["commitment_loss"],
                    f"{prefix}/temporal_smooth_loss": teacher_plan["temporal_smooth_loss"],
                },
                prog_bar=True,
                batch_size=batch["pixels"].shape[0],
            )
            return loss

        teacher_plan = self._teacher_plan(batch, detach=self.stage_name != "tokenizer")

        if self.stage_name == "planner":
            plan_loss, plan_acc = self._plan_loss(current, teacher_plan)
            self.log_dict(
                {
                    f"{prefix}/loss": plan_loss,
                    f"{prefix}_loss": plan_loss,
                    f"{prefix}/plan_loss": plan_loss,
                    f"{prefix}/token_acc": plan_acc,
                },
                prog_bar=True,
                batch_size=batch["pixels"].shape[0],
            )
            return plan_loss

        plan_embeddings, plan_loss, plan_acc = self._plan_for_conditioning(
            current,
            teacher_plan,
            batch_size=batch["pixels"].shape[0],
        )

        losses = self._action_and_aux_losses(
            current,
            plan_embeddings=plan_embeddings,
            teacher_plan=teacher_plan,
            target_actions=target_actions,
        )
        total = (
            float(self.cfg.loss.planner_weight) * plan_loss
            + float(self.cfg.loss.action_weight) * losses["action_loss"]
            + float(self.cfg.loss.consistency_weight) * losses["consistency_loss"]
            + float(self.cfg.loss.video_weight) * losses["video_loss"]
        )
        self.log_dict(
            {
                f"{prefix}/loss": total,
                f"{prefix}_loss": total,
                f"{prefix}/plan_loss": plan_loss,
                f"{prefix}/token_acc": plan_acc,
                f"{prefix}/action_loss": losses["action_loss"],
                f"{prefix}/action_mse": losses["action_mse"],
                f"{prefix}/consistency_loss": losses["consistency_loss"],
                f"{prefix}/video_loss": losses["video_loss"],
                f"{prefix}/teacher_ratio": torch.tensor(self._teacher_ratio(), device=self.device),
            },
            prog_bar=True,
            batch_size=batch["pixels"].shape[0],
        )
        return total

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        return optimizer
