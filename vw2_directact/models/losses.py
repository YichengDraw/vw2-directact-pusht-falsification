from __future__ import annotations

import torch
import torch.nn.functional as F


def action_huber_loss(prediction: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(prediction, target, delta=delta)


def cosine_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    mse = F.mse_loss(prediction, target)
    return cosine + mse


def info_nce_loss(query: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    normalized_query = F.normalize(query, dim=-1)
    normalized_target = F.normalize(target, dim=-1)
    logits = normalized_query @ normalized_target.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(query.shape[0], device=query.device)
    return F.cross_entropy(logits, labels)


def token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (torch.argmax(logits, dim=-1) == target).float().mean()


def vicreg_variance_covariance_loss(
    embedding: torch.Tensor,
    *,
    variance_floor: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if embedding.shape[0] <= 1:
        zero = embedding.new_zeros(())
        return zero, zero, zero

    centered = embedding - embedding.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    variance_loss = F.relu(variance_floor - std).mean()

    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(embedding.shape[0] - 1, 1)
    eye = torch.eye(covariance.shape[0], device=embedding.device, dtype=torch.bool)
    off_diagonal = covariance.masked_select(~eye)
    covariance_loss = off_diagonal.pow(2).mean() if off_diagonal.numel() > 0 else embedding.new_zeros(())
    return variance_loss + covariance_loss, variance_loss, covariance_loss
