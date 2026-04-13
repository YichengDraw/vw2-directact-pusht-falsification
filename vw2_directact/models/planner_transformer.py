from __future__ import annotations

import torch
from torch import nn


class PlannerTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        plan_tokens: int,
        use_vq: bool,
        codebook_size: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.plan_tokens = plan_tokens
        self.use_vq = use_vq
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.obs_proj = nn.Linear(hidden_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(1, plan_tokens + 1, hidden_dim) * 0.02)
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
        self.input_proj = nn.Embedding(codebook_size, hidden_dim) if use_vq else nn.Linear(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, codebook_size if use_vq else hidden_dim)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def forward_train(self, obs_summary: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_embeddings = self.input_proj(targets)
        history = torch.cat([self.start_token.expand(obs_summary.shape[0], -1, -1), target_embeddings[:, :-1]], dim=1)
        history = history + self.position[:, : history.shape[1]]
        history[:, 0] = history[:, 0] + self.obs_proj(obs_summary)
        hidden = self.transformer(history, mask=self._causal_mask(history.shape[1], history.device))
        return self.output_head(hidden)

    def generate(self, obs_summary: torch.Tensor, *, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        history = self.start_token.expand(obs_summary.shape[0], 1, -1)
        logits_steps: list[torch.Tensor] = []
        token_ids: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []

        for _ in range(self.plan_tokens):
            sequence = history + self.position[:, : history.shape[1]]
            sequence[:, 0] = sequence[:, 0] + self.obs_proj(obs_summary)
            hidden = self.transformer(sequence, mask=self._causal_mask(sequence.shape[1], sequence.device))
            next_output = self.output_head(hidden[:, -1])
            logits_steps.append(next_output)
            if self.use_vq:
                next_token = torch.argmax(next_output / max(temperature, 1e-6), dim=-1)
                token_ids.append(next_token)
                next_embedding = self.input_proj(next_token).unsqueeze(1)
            else:
                latents.append(next_output)
                next_embedding = self.input_proj(next_output).unsqueeze(1)
            history = torch.cat([history, next_embedding], dim=1)

        result = {"logits": torch.stack(logits_steps, dim=1)}
        if self.use_vq:
            result["token_ids"] = torch.stack(token_ids, dim=1)
        else:
            result["latents"] = torch.stack(latents, dim=1)
        return result
