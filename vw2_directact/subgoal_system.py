from __future__ import annotations

from contextlib import nullcontext

import lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .data import PushTSubgoalDataset
from .data.common import build_torch_split
from .models import ActionDecoder, FutureBottleneck, HistoryEncoder, ObservationEncoder, SubgoalPredictor
from .models.losses import (
    action_huber_loss,
    cosine_mse_loss,
    info_nce_loss,
    vicreg_variance_covariance_loss,
)
from .utils.metrics import (
    batch_action_mse,
    covariance_offdiag_mean,
    mean_feature_variance,
    retrieval_top1,
    shuffled_retrieval_top1,
)

_STAGE_KIND = {
    "teacher": "teacher",
    "teacher_oracle": "teacher",
    "student": "student",
    "student_predictor": "student",
    "joint": "joint",
    "joint_subgoal": "joint",
}


class VW2SubgoalDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def _make_dataset(self, *, train: bool) -> PushTSubgoalDataset:
        if str(self.cfg.data.dataset_type) != "pusht":
            raise ValueError("The subgoal-distillation pipeline currently supports Push-T only.")
        max_samples = self.cfg.data.max_train_samples if train else self.cfg.data.max_val_samples
        return PushTSubgoalDataset(
            path=self.cfg.data.path,
            dataset_name=self.cfg.data.dataset_name,
            cache_dir=self.cfg.data.cache_dir,
            image_size=int(self.cfg.data.image_size),
            history_steps=int(self.cfg.subgoal.history_steps),
            future_horizon=int(self.cfg.data.plan_horizon),
            action_horizon=int(self.cfg.model.action_chunk),
            train=train,
            train_split=float(self.cfg.data.train_split),
            seed=int(self.cfg.seed),
            stride=int(self.cfg.data.stride),
            max_samples=None if max_samples is None else int(max_samples),
        )

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


class VW2SubgoalModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.subgoal_dim = int(cfg.subgoal.subgoal_dim)
        self.action_dim = int(cfg.model.action_dim)
        self.action_chunk = int(cfg.model.action_chunk)
        self.future_horizon = int(cfg.data.plan_horizon)
        self.history_steps = int(cfg.subgoal.history_steps)
        self.obs_encoder = ObservationEncoder(
            hidden_dim=int(cfg.model.hidden_dim),
            image_channels=int(cfg.model.image_channels),
            proprio_dim=int(cfg.model.proprio_dim),
            language_dim=int(cfg.model.language_dim),
            freeze_encoder=bool(cfg.model.freeze_encoder),
        )
        self.history_encoder = HistoryEncoder(
            hidden_dim=int(cfg.model.hidden_dim),
            action_dim=int(cfg.model.action_dim),
            history_steps=int(cfg.subgoal.history_steps),
            layers=int(cfg.subgoal.history_layers),
            heads=int(cfg.subgoal.history_heads),
        )
        self.future_bottleneck = FutureBottleneck(
            hidden_dim=int(cfg.model.hidden_dim),
            subgoal_dim=int(cfg.subgoal.subgoal_dim),
            future_horizon=int(cfg.data.plan_horizon),
            layers=int(cfg.subgoal.future_layers),
            heads=int(cfg.subgoal.future_heads),
        )
        self.subgoal_predictor = SubgoalPredictor(
            hidden_dim=int(cfg.model.hidden_dim),
            subgoal_dim=int(cfg.subgoal.subgoal_dim),
            max_horizon=int(cfg.subgoal.max_horizon),
        )
        self.subgoal_projection = nn.Sequential(
            nn.LayerNorm(int(cfg.subgoal.subgoal_dim)),
            nn.Linear(int(cfg.subgoal.subgoal_dim), int(cfg.model.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(cfg.model.hidden_dim), int(cfg.model.hidden_dim)),
        )
        self.action_decoder = ActionDecoder(
            hidden_dim=int(cfg.model.hidden_dim),
            action_dim=int(cfg.model.action_dim),
            action_chunk=int(cfg.model.action_chunk),
            layers=int(cfg.model.action_decoder_layers),
            heads=int(cfg.model.action_decoder_heads),
        )

    def encode_history(
        self,
        *,
        history_pixels: torch.Tensor,
        history_proprio: torch.Tensor | None,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.obs_encoder.encode_sequence(
            pixels=history_pixels,
            proprio=history_proprio,
        )
        return self.history_encoder(
            observation_sequence=encoded["summary_sequence"],
            previous_actions=prev_actions,
        )

    def encode_future_subgoal(
        self,
        *,
        future_pixels: torch.Tensor,
        future_proprio: torch.Tensor | None,
    ) -> torch.Tensor:
        encoded = self.obs_encoder.encode_sequence(
            pixels=future_pixels,
            proprio=future_proprio,
        )
        return self.future_bottleneck(future_sequence=encoded["summary_sequence"])

    def predict_subgoal(self, *, context: torch.Tensor, horizon_steps: torch.Tensor | int) -> torch.Tensor:
        return self.subgoal_predictor(context=context, horizon_steps=horizon_steps)

    def act(self, *, context: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        subgoal_token = self.subgoal_projection(subgoal).unsqueeze(1)
        return self.action_decoder(
            obs_summary=context,
            plan_tokens=subgoal_token,
            proprio_token=None,
            language_token=None,
        )

    def zero_subgoal(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.subgoal_dim, device=device)

    def predict_action_chunk(
        self,
        *,
        history_pixels: torch.Tensor,
        history_proprio: torch.Tensor | None,
        prev_actions: torch.Tensor,
        horizon_steps: torch.Tensor | int | None = None,
        mode: str = "student",
        oracle_subgoal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encode_history(
            history_pixels=history_pixels,
            history_proprio=history_proprio,
            prev_actions=prev_actions,
        )
        if horizon_steps is None:
            horizon_steps = self.future_horizon
        if mode == "bc":
            subgoal = self.zero_subgoal(context.shape[0], device=context.device)
        elif mode == "oracle":
            if oracle_subgoal is None:
                raise ValueError("Oracle mode requires oracle_subgoal.")
            subgoal = oracle_subgoal
        elif mode in {"student", "predfuture"}:
            subgoal = self.predict_subgoal(context=context, horizon_steps=horizon_steps)
        else:
            raise ValueError(f"Unsupported subgoal policy mode={mode!r}.")
        return self.act(context=context, subgoal=subgoal)


class VW2SubgoalSystem(pl.LightningModule):
    def __init__(self, cfg: DictConfig, stage_name: str) -> None:
        super().__init__()
        if stage_name not in _STAGE_KIND:
            raise ValueError(f"Unsupported stage_name={stage_name!r}.")
        self.cfg = cfg
        self.stage_name = stage_name
        self.stage_kind = _STAGE_KIND[stage_name]
        self.model = VW2SubgoalModel(cfg)
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True), "stage_name": stage_name})
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        for module in (
            self.model.obs_encoder,
            self.model.history_encoder,
            self.model.future_bottleneck,
            self.model.subgoal_predictor,
            self.model.subgoal_projection,
            self.model.action_decoder,
        ):
            module.requires_grad_(False)

        if self.stage_kind == "teacher":
            self.model.obs_encoder.requires_grad_(not bool(self.cfg.model.freeze_encoder))
            self.model.history_encoder.requires_grad_(True)
            self.model.future_bottleneck.requires_grad_(True)
            self.model.subgoal_projection.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            return

        if self.stage_kind == "student":
            self.model.subgoal_predictor.requires_grad_(True)
            return

        self.model.obs_encoder.requires_grad_(not bool(self.cfg.model.freeze_encoder))
        self.model.history_encoder.requires_grad_(True)
        self.model.future_bottleneck.requires_grad_(True)
        self.model.subgoal_predictor.requires_grad_(True)
        self.model.subgoal_projection.requires_grad_(True)
        self.model.action_decoder.requires_grad_(True)

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

    def _encode_context(self, batch: dict[str, torch.Tensor], *, detach: bool) -> torch.Tensor:
        context = torch.no_grad if detach else nullcontext
        with context():
            encoded = self.model.encode_history(
                history_pixels=batch["history_pixels"],
                history_proprio=batch.get("history_proprio"),
                prev_actions=batch["prev_actions"],
            )
        return encoded.detach() if detach else encoded

    def _teacher_subgoal(self, batch: dict[str, torch.Tensor], *, detach: bool) -> torch.Tensor:
        context = torch.no_grad if detach else nullcontext
        with context():
            subgoal = self.model.encode_future_subgoal(
                future_pixels=batch["future_pixels"],
                future_proprio=batch.get("future_proprio"),
            )
        return subgoal.detach() if detach else subgoal

    def _horizon_steps(self, batch: dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        if "horizon" in batch:
            return batch["horizon"].to(device=self.device, dtype=torch.long)
        return torch.full((batch_size,), int(self.cfg.data.plan_horizon), device=self.device, dtype=torch.long)

    def _diagnostics(
        self,
        *,
        predicted_subgoal: torch.Tensor,
        teacher_subgoal: torch.Tensor,
        predicted_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "subgoal_batch_variance": mean_feature_variance(predicted_subgoal),
            "covariance_offdiag_mean": covariance_offdiag_mean(predicted_subgoal),
            "retrieval_top1": retrieval_top1(predicted_subgoal, teacher_subgoal),
            "retrieval_top1_shuffled": shuffled_retrieval_top1(predicted_subgoal, teacher_subgoal),
            "action_gap": batch_action_mse(predicted_actions, teacher_actions),
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="val")

    def _shared_step(self, batch: dict[str, torch.Tensor], *, prefix: str) -> torch.Tensor:
        batch_size = batch["history_pixels"].shape[0]
        delta = float(self.cfg.loss.huber_delta)
        horizon_steps = self._horizon_steps(batch, batch_size)
        target_actions = batch["action"][:, : int(self.cfg.model.action_chunk)]

        if self.stage_kind == "teacher":
            context = self._encode_context(batch, detach=False)
            teacher_subgoal = self._teacher_subgoal(batch, detach=False)
            teacher_actions = self.model.act(context=context, subgoal=teacher_subgoal)
            teacher_action_loss = action_huber_loss(teacher_actions, target_actions, delta=delta)
            diagnostics = {
                "subgoal_batch_variance": mean_feature_variance(teacher_subgoal),
                "covariance_offdiag_mean": covariance_offdiag_mean(teacher_subgoal),
            }
            self.log_dict(
                {
                    f"{prefix}/loss": teacher_action_loss,
                    f"{prefix}_loss": teacher_action_loss,
                    f"{prefix}/teacher_action_loss": teacher_action_loss,
                    f"{prefix}/teacher_action_mse": F.mse_loss(teacher_actions, target_actions),
                    f"{prefix}/subgoal_batch_variance": diagnostics["subgoal_batch_variance"],
                    f"{prefix}/covariance_offdiag_mean": diagnostics["covariance_offdiag_mean"],
                },
                prog_bar=True,
                batch_size=batch_size,
            )
            return teacher_action_loss

        context = self._encode_context(batch, detach=self.stage_kind == "student")
        teacher_subgoal = self._teacher_subgoal(batch, detach=self.stage_kind == "student")
        teacher_actions = self.model.act(context=context, subgoal=teacher_subgoal)
        predicted_subgoal = self.model.predict_subgoal(context=context, horizon_steps=horizon_steps)
        predicted_actions = self.model.act(context=context, subgoal=predicted_subgoal)

        teacher_action_loss = action_huber_loss(teacher_actions, target_actions, delta=delta)
        subgoal_loss = cosine_mse_loss(predicted_subgoal, teacher_subgoal)
        actdistill_loss = action_huber_loss(predicted_actions, teacher_actions.detach(), delta=delta)
        vicreg_loss, variance_loss, covariance_loss = vicreg_variance_covariance_loss(
            predicted_subgoal,
            variance_floor=float(self.cfg.loss.subgoal_variance_floor),
        )
        nce_loss = info_nce_loss(
            predicted_subgoal,
            teacher_subgoal.detach() if self.stage_kind == "joint" else teacher_subgoal,
            temperature=float(self.cfg.loss.nce_temperature),
        )
        diagnostics = self._diagnostics(
            predicted_subgoal=predicted_subgoal,
            teacher_subgoal=teacher_subgoal.detach(),
            predicted_actions=predicted_actions,
            teacher_actions=teacher_actions.detach(),
        )

        total = (
            float(self.cfg.loss.subgoal_weight) * subgoal_loss
            + float(self.cfg.loss.actdistill_weight) * actdistill_loss
            + float(self.cfg.loss.var_weight) * vicreg_loss
            + float(self.cfg.loss.nce_weight) * nce_loss
        )
        if self.stage_kind == "joint":
            total = total + float(self.cfg.loss.action_weight) * teacher_action_loss

        self.log_dict(
            {
                f"{prefix}/loss": total,
                f"{prefix}_loss": total,
                f"{prefix}/teacher_action_loss": teacher_action_loss,
                f"{prefix}/teacher_action_mse": F.mse_loss(teacher_actions, target_actions),
                f"{prefix}/student_action_mse": F.mse_loss(predicted_actions, target_actions),
                f"{prefix}/subgoal_loss": subgoal_loss,
                f"{prefix}/actdistill_loss": actdistill_loss,
                f"{prefix}/var_loss": variance_loss,
                f"{prefix}/cov_loss": covariance_loss,
                f"{prefix}/nce_loss": nce_loss,
                f"{prefix}/subgoal_batch_variance": diagnostics["subgoal_batch_variance"],
                f"{prefix}/covariance_offdiag_mean": diagnostics["covariance_offdiag_mean"],
                f"{prefix}/retrieval_top1": diagnostics["retrieval_top1"],
                f"{prefix}/retrieval_top1_shuffled": diagnostics["retrieval_top1_shuffled"],
                f"{prefix}/action_gap": diagnostics["action_gap"],
            },
            prog_bar=True,
            batch_size=batch_size,
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
