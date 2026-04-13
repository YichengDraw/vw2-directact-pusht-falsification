from __future__ import annotations

import torch
from torch import nn


class MLPTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class ConvImageTokenizer(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return features.flatten(2).transpose(1, 2)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        image_channels: int,
        proprio_dim: int = 0,
        language_dim: int = 0,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_encoder = ConvImageTokenizer(image_channels, hidden_dim)
        self.proprio_encoder = MLPTokenEncoder(proprio_dim, hidden_dim) if proprio_dim > 0 else None
        self.language_encoder = MLPTokenEncoder(language_dim, hidden_dim) if language_dim > 0 else None
        self.static_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gripper_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.proprio_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.language_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

        if freeze_encoder:
            self.image_encoder.requires_grad_(False)

    def _image_tokens(self, image: torch.Tensor | None, token_type: torch.Tensor) -> torch.Tensor | None:
        if image is None:
            return None
        return self.image_encoder(image) + token_type

    def encode_observation(
        self,
        *,
        pixels: torch.Tensor | None,
        gripper_pixels: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        pieces: list[torch.Tensor] = []
        static_tokens = self._image_tokens(pixels, self.static_type)
        if static_tokens is not None:
            pieces.append(static_tokens)

        gripper_tokens = self._image_tokens(gripper_pixels, self.gripper_type)
        if gripper_tokens is not None:
            pieces.append(gripper_tokens)

        proprio_token = None
        if proprio is not None and self.proprio_encoder is not None:
            proprio_token = self.proprio_encoder(proprio).unsqueeze(1) + self.proprio_type
            pieces.append(proprio_token)

        language_token = None
        if language is not None and self.language_encoder is not None:
            language_token = self.language_encoder(language).unsqueeze(1) + self.language_type
            pieces.append(language_token)

        if not pieces:
            raise ValueError("At least one modality is required to encode an observation.")

        tokens = self.norm(torch.cat(pieces, dim=1))
        summary = tokens.mean(dim=1)
        return {
            "tokens": tokens,
            "summary": summary,
            "proprio_token": None if proprio_token is None else proprio_token.squeeze(1),
            "language_token": None if language_token is None else language_token.squeeze(1),
        }

    def encode_sequence(
        self,
        *,
        pixels: torch.Tensor | None,
        gripper_pixels: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        reference = pixels if pixels is not None else gripper_pixels
        if reference is None:
            raise ValueError("Sequence encoding requires pixel inputs.")

        batch_size, time_steps = reference.shape[:2]

        def flatten(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.reshape(batch_size * time_steps, *value.shape[2:])

        encoded = self.encode_observation(
            pixels=flatten(pixels),
            gripper_pixels=flatten(gripper_pixels),
            proprio=flatten(proprio),
            language=flatten(language),
        )

        tokens = encoded["tokens"].reshape(batch_size, time_steps, encoded["tokens"].shape[1], self.hidden_dim)
        summary = encoded["summary"].reshape(batch_size, time_steps, self.hidden_dim)
        result: dict[str, torch.Tensor | None] = {"tokens": tokens, "summary_sequence": summary}
        for key in ("proprio_token", "language_token"):
            value = encoded[key]
            result[key] = None if value is None else value.reshape(batch_size, time_steps, self.hidden_dim)
        return result
