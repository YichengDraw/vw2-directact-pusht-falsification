from __future__ import annotations

import torch
from torch import nn


class SubgoalPredictor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        subgoal_dim: int,
        max_horizon: int,
    ) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.horizon_embedding = nn.Embedding(max_horizon + 1, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, subgoal_dim),
        )

    def forward(self, *, context: torch.Tensor, horizon_steps: torch.Tensor | int) -> torch.Tensor:
        if not torch.is_tensor(horizon_steps):
            horizon_steps = torch.full((context.shape[0],), int(horizon_steps), device=context.device, dtype=torch.long)
        horizon_steps = horizon_steps.to(device=context.device, dtype=torch.long).clamp_(0, self.max_horizon)
        horizon_tokens = self.horizon_embedding(horizon_steps)
        return self.net(torch.cat([context, horizon_tokens], dim=-1))
