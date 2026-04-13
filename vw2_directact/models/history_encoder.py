from __future__ import annotations

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        history_steps: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(1, history_steps, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, *, observation_sequence: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        if observation_sequence.shape[:2] != previous_actions.shape[:2]:
            raise ValueError("Observation history and previous action history must share batch/time dimensions.")
        hidden = observation_sequence + self.action_proj(previous_actions) + self.position[:, : observation_sequence.shape[1]]
        hidden = self.transformer(hidden)
        pooled = hidden[:, -1] + hidden.mean(dim=1)
        return self.head(pooled)
