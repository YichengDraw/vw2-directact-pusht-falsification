from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def batch_action_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).mean()


def mean_feature_variance(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] <= 1:
        return embedding.new_zeros(())
    return embedding.var(dim=0, unbiased=False).mean()


def covariance_offdiag_mean(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] <= 1:
        return embedding.new_zeros(())
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(embedding.shape[0] - 1, 1)
    eye = torch.eye(covariance.shape[0], device=embedding.device, dtype=torch.bool)
    off_diagonal = covariance.masked_select(~eye)
    return off_diagonal.abs().mean() if off_diagonal.numel() > 0 else embedding.new_zeros(())


def retrieval_top1(query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if query.shape[0] == 0:
        return query.new_zeros(())
    similarity = F.normalize(query, dim=-1) @ F.normalize(target, dim=-1).transpose(0, 1)
    labels = torch.arange(query.shape[0], device=query.device)
    return (similarity.argmax(dim=1) == labels).float().mean()


def shuffled_retrieval_top1(query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if query.shape[0] <= 1:
        return query.new_zeros(())
    permutation = torch.roll(torch.arange(query.shape[0], device=query.device), shifts=1)
    shuffled_target = target[permutation]
    similarity = F.normalize(query, dim=-1) @ F.normalize(shuffled_target, dim=-1).transpose(0, 1)
    labels = torch.arange(query.shape[0], device=query.device)
    return (similarity.argmax(dim=1) == labels).float().mean()


def latency_ms(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0
