from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..data.common import IMAGENET_MEAN, IMAGENET_STD


def _last_step(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if tensor.ndim >= 3:
        return tensor[:, -1]
    return tensor


def _prepare_pixels(value: Any, *, image_size: int, device: torch.device) -> torch.Tensor | None:
    tensor = _last_step(value)
    if tensor is None:
        return None
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.float().to(device)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    if tensor.shape[-1] != image_size or tensor.shape[-2] != image_size:
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(device=device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(device=device, dtype=tensor.dtype)
    return (tensor - mean) / std


def _prepare_vector(value: Any, *, device: torch.device) -> torch.Tensor | None:
    tensor = _last_step(value)
    if tensor is None:
        return None
    return tensor.float().to(device)


def prepare_policy_batch(info_dict: dict[str, Any], *, image_size: int, device: torch.device) -> dict[str, torch.Tensor | None]:
    return {
        "pixels": _prepare_pixels(info_dict.get("pixels"), image_size=image_size, device=device),
        "gripper_pixels": _prepare_pixels(info_dict.get("gripper_pixels"), image_size=image_size, device=device),
        "proprio": _prepare_vector(info_dict.get("proprio"), device=device),
        "language": _prepare_vector(info_dict.get("language"), device=device),
    }


try:
    from stable_worldmodel.policy import BasePolicy
except Exception:  # pragma: no cover
    BasePolicy = object


class DirectActPolicy(BasePolicy):
    def __init__(
        self,
        *,
        model,
        image_size: int,
        execute_steps: int,
        mode: str,
        temperature: float = 1.0,
        oracle_plan_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model.eval()
        self.image_size = image_size
        self.execute_steps = execute_steps
        self.mode = mode
        self.temperature = temperature
        self.oracle_plan_embeddings = oracle_plan_embeddings
        self._action_queue: torch.Tensor | None = None
        self._steps_until_replan = 0

    def get_action(self, info_dict, **kwargs):
        device = next(self.model.parameters()).device
        if self._action_queue is None or self._steps_until_replan <= 0:
            batch = prepare_policy_batch(info_dict, image_size=self.image_size, device=device)
            plan_override = None
            if self.mode == "oracle":
                if self.oracle_plan_embeddings is None:
                    raise ValueError("Oracle policy mode requires oracle_plan_embeddings.")
                plan_override = self.oracle_plan_embeddings.to(device)
            with torch.no_grad():
                predicted = self.model.predict_action_chunk(
                    pixels=batch["pixels"],
                    gripper_pixels=batch["gripper_pixels"],
                    proprio=batch["proprio"],
                    language=batch["language"],
                    temperature=self.temperature,
                    mode=self.mode,
                    plan_override=plan_override,
                )
            self._action_queue = predicted.detach().cpu()
            self._steps_until_replan = self.execute_steps

        action = self._action_queue[:, 0].numpy()
        self._action_queue = self._action_queue[:, 1:] if self._action_queue.shape[1] > 1 else None
        self._steps_until_replan -= 1
        return action


class SubgoalPolicy(BasePolicy):
    def __init__(
        self,
        *,
        model,
        image_size: int,
        history_steps: int,
        action_dim: int,
        execute_steps: int,
        horizon_steps: int,
        mode: str,
        oracle_subgoal: torch.Tensor | None = None,
        bootstrap_history_pixels: torch.Tensor | None = None,
        bootstrap_history_proprio: torch.Tensor | None = None,
        bootstrap_prev_actions: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model.eval()
        self.image_size = image_size
        self.history_steps = history_steps
        self.action_dim = action_dim
        self.execute_steps = execute_steps
        self.horizon_steps = horizon_steps
        self.mode = mode
        self.oracle_subgoal = oracle_subgoal
        self._action_queue: torch.Tensor | None = None
        self._steps_until_replan = 0
        self._history_pixels = bootstrap_history_pixels
        self._history_proprio = bootstrap_history_proprio
        self._prev_actions = bootstrap_prev_actions
        self._skip_first_observation = bootstrap_history_pixels is not None

    def _repeat_history(self, value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(1).repeat_interleave(self.history_steps, dim=1)

    def _update_history(self, *, pixels: torch.Tensor, proprio: torch.Tensor | None) -> None:
        if self._history_pixels is None:
            self._history_pixels = self._repeat_history(pixels)
            self._history_proprio = None if proprio is None else self._repeat_history(proprio)
            device = pixels.device
            self._prev_actions = torch.zeros(pixels.shape[0], self.history_steps, self.action_dim, device=device)
            return

        if self._skip_first_observation:
            self._skip_first_observation = False
            return

        self._history_pixels = torch.cat([self._history_pixels[:, 1:], pixels.unsqueeze(1)], dim=1)
        if self._history_proprio is not None and proprio is not None:
            self._history_proprio = torch.cat([self._history_proprio[:, 1:], proprio.unsqueeze(1)], dim=1)
        elif self._history_proprio is None and proprio is not None:
            self._history_proprio = self._repeat_history(proprio)

    def _append_action(self, action: torch.Tensor) -> None:
        if self._prev_actions is None:
            self._prev_actions = torch.zeros(action.shape[0], self.history_steps, self.action_dim, device=action.device)
        self._prev_actions = torch.cat([self._prev_actions[:, 1:], action.unsqueeze(1)], dim=1)

    def get_action(self, info_dict, **kwargs):
        device = next(self.model.parameters()).device
        pixels = _prepare_pixels(info_dict.get("pixels"), image_size=self.image_size, device=device)
        proprio = _prepare_vector(info_dict.get("proprio"), device=device)
        if pixels is None:
            raise ValueError("SubgoalPolicy requires pixel observations.")
        self._update_history(pixels=pixels, proprio=proprio)

        if self._action_queue is None or self._steps_until_replan <= 0:
            horizon = torch.full((pixels.shape[0],), self.horizon_steps, device=device, dtype=torch.long)
            oracle_subgoal = None if self.oracle_subgoal is None else self.oracle_subgoal.to(device)
            with torch.no_grad():
                predicted = self.model.predict_action_chunk(
                    history_pixels=self._history_pixels,
                    history_proprio=self._history_proprio,
                    prev_actions=self._prev_actions,
                    horizon_steps=horizon,
                    mode=self.mode,
                    oracle_subgoal=oracle_subgoal,
                )
            self._action_queue = predicted.detach().cpu()
            self._steps_until_replan = self.execute_steps

        action = self._action_queue[:, 0].numpy()
        self._action_queue = self._action_queue[:, 1:] if self._action_queue.shape[1] > 1 else None
        self._steps_until_replan -= 1
        self._append_action(torch.as_tensor(action, device=device, dtype=torch.float32))
        return action
