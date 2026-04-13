from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        embedding = torch.randn(codebook_size, embedding_dim)
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embedding_avg", embedding.clone())

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = inputs.reshape(-1, self.embedding_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_size).type_as(flat)
        quantized = F.embedding(encoding_indices, self.embedding).view_as(inputs)

        if self.training:
            counts = encodings.sum(dim=0)
            embeds = encodings.transpose(0, 1) @ flat
            self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.embedding_avg.mul_(self.decay).add_(embeds, alpha=1 - self.decay)
            total = self.cluster_size.sum()
            normalized = (self.cluster_size + self.eps) / (total + self.codebook_size * self.eps) * total
            self.embedding.copy_(self.embedding_avg / normalized.unsqueeze(1))

        commitment_loss = F.mse_loss(inputs, quantized.detach())
        quantized = inputs + (quantized - inputs).detach()
        return {
            "quantized": quantized,
            "token_ids": encoding_indices.view(*inputs.shape[:-1]),
            "commitment_loss": commitment_loss,
        }

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.embedding)


class TemporalDynamicsTokenizer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        chunk_horizon: int,
        num_queries: int,
        use_vq: bool,
        codebook_size: int,
        heads: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_horizon = chunk_horizon
        self.num_queries = num_queries
        self.vq = VectorQuantizerEMA(codebook_size, hidden_dim) if use_vq else None
        self.position = nn.Parameter(torch.randn(1, chunk_horizon, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_queries),
            nn.Linear(hidden_dim * num_queries, hidden_dim * chunk_horizon),
        )

    def forward(self, future_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        if future_summary.ndim == 3:
            future_summary = future_summary.unsqueeze(1)

        batch_size, num_chunks, chunk_horizon, hidden_dim = future_summary.shape
        if chunk_horizon != self.chunk_horizon:
            raise ValueError(f"Expected horizon {self.chunk_horizon}, got {chunk_horizon}.")

        flat = future_summary.reshape(batch_size * num_chunks, chunk_horizon, hidden_dim)
        encoded = self.temporal_encoder(flat + self.position[:, :chunk_horizon])
        queries = self.query_tokens.expand(flat.shape[0], -1, -1)
        pooled, _ = self.cross_attention(queries, encoded, encoded)

        commit_loss = torch.zeros((), device=future_summary.device)
        token_ids = torch.empty(0, dtype=torch.long, device=future_summary.device)
        plan_embeddings = pooled
        if self.vq is not None:
            vq_output = self.vq(pooled)
            plan_embeddings = vq_output["quantized"]
            token_ids = vq_output["token_ids"]
            commit_loss = vq_output["commitment_loss"]

        recon = self.reconstruction_head(plan_embeddings.reshape(flat.shape[0], -1)).view(flat.shape[0], chunk_horizon, hidden_dim)
        return {
            "plan_embeddings": plan_embeddings.reshape(batch_size, num_chunks * self.num_queries, hidden_dim),
            "token_ids": token_ids.reshape(batch_size, num_chunks * self.num_queries) if token_ids.numel() > 0 else token_ids,
            "reconstruction": recon.reshape(batch_size, num_chunks, chunk_horizon, hidden_dim),
            "recon_loss": F.mse_loss(recon, flat),
            "commitment_loss": commit_loss,
            "temporal_smooth_loss": F.mse_loss(recon[:, 1:] - recon[:, :-1], flat[:, 1:] - flat[:, :-1]),
        }

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.vq is None:
            raise RuntimeError("Codebook lookup is only available with VQ enabled.")
        return self.vq.lookup(token_ids)
