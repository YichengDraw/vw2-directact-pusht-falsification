from __future__ import annotations

import torch
from torch import nn


class FutureBottleneck(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        subgoal_dim: int,
        future_horizon: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, future_horizon, hidden_dim) * 0.02)
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
            nn.Linear(hidden_dim, subgoal_dim),
        )

    def forward(self, *, future_sequence: torch.Tensor) -> torch.Tensor:
        hidden = future_sequence + self.position[:, : future_sequence.shape[1]]
        hidden = self.transformer(hidden)
        return self.head(hidden.mean(dim=1))
