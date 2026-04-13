from __future__ import annotations

import torch
from torch import nn


class ForwardConsistencyModel(nn.Module):
    def __init__(self, *, hidden_dim: int, action_dim: int, action_chunk: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim * action_chunk, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, obs_summary: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs_summary, action_chunk.reshape(action_chunk.shape[0], -1)], dim=-1))


class FutureFeatureHead(nn.Module):
    def __init__(self, *, hidden_dim: int, plan_horizon: int) -> None:
        super().__init__()
        self.plan_horizon = plan_horizon
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * plan_horizon),
        )

    def forward(self, obs_summary: torch.Tensor, plan_tokens: torch.Tensor) -> torch.Tensor:
        plan_summary = plan_tokens.mean(dim=1) if plan_tokens.numel() > 0 else torch.zeros_like(obs_summary)
        output = self.net(torch.cat([obs_summary, plan_summary], dim=-1))
        return output.view(obs_summary.shape[0], self.plan_horizon, -1)
