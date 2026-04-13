from __future__ import annotations

import torch
from torch import nn


class ActionDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        action_chunk: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.action_chunk = action_chunk
        self.action_queries = nn.Parameter(torch.randn(1, action_chunk, hidden_dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
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
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        *,
        obs_summary: torch.Tensor,
        plan_tokens: torch.Tensor | None,
        proprio_token: torch.Tensor | None = None,
        language_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = [obs_summary.unsqueeze(1)]
        if proprio_token is not None:
            context.append(proprio_token.unsqueeze(1))
        if language_token is not None:
            context.append(language_token.unsqueeze(1))
        if plan_tokens is not None and plan_tokens.numel() > 0:
            context.append(plan_tokens)
        sequence = torch.cat(context + [self.action_queries.expand(obs_summary.shape[0], -1, -1)], dim=1)
        sequence = sequence + self.position[:, : sequence.shape[1]]
        hidden = self.transformer(sequence)
        return self.head(hidden[:, -self.action_chunk :])
