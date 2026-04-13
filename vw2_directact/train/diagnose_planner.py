from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..models.losses import token_accuracy
from ..system import VW2DirectActDataModule, VW2DirectActSystem
from .common import load_cfg_for_eval


def _to_device(batch, device):
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def main() -> None:
    args, cfg = load_cfg_for_eval()
    system = VW2DirectActSystem(cfg, "joint")
    system.load_weights_from_checkpoint(args.checkpoint)
    if not bool(cfg.model.use_vq):
        raise ValueError("Planner diagnostics require model.use_vq=true.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system.to(device)
    system.eval()

    datamodule = VW2DirectActDataModule(cfg, "joint")
    datamodule.setup()
    loader = datamodule.val_dataloader()

    entropy_values = []
    predicted_token_ids = []
    normal_acc = []
    shuffled_acc = []
    zero_obs_acc = []

    for batch in loader:
        batch = _to_device(batch, device)
        with torch.no_grad():
            current = system._encode_current(batch)
            teacher_plan = system._teacher_plan(batch, detach=True)
            logits = system.model.planner.forward_train(current.summary, teacher_plan["token_ids"])
            normal_acc.append(float(token_accuracy(logits, teacher_plan["token_ids"]).cpu()))

            if teacher_plan["token_ids"].shape[0] > 1:
                permutation = torch.randperm(teacher_plan["token_ids"].shape[0], device=device)
                shuffled_targets = teacher_plan["token_ids"][permutation]
            else:
                shuffled_targets = teacher_plan["token_ids"]
            shuffled_acc.append(float(token_accuracy(logits, shuffled_targets).cpu()))

            zero_logits = system.model.planner.forward_train(torch.zeros_like(current.summary), teacher_plan["token_ids"])
            zero_obs_acc.append(float(token_accuracy(zero_logits, teacher_plan["token_ids"]).cpu()))

            generated = system.model.planner.generate(current.summary, temperature=float(cfg.sampling.temperature))
            probabilities = torch.softmax(generated["logits"], dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
            entropy_values.append(entropy.cpu().reshape(-1))
            predicted_token_ids.append(generated["token_ids"].cpu().reshape(-1))

    if not predicted_token_ids:
        raise RuntimeError("No batches were processed for planner diagnostics.")

    all_token_ids = torch.cat(predicted_token_ids)
    histogram = torch.bincount(all_token_ids, minlength=int(cfg.model.codebook_size)).float()
    token_probs = histogram / histogram.sum().clamp_min(1.0)
    codebook_entropy = -(token_probs[token_probs > 0] * token_probs[token_probs > 0].log()).sum()

    diagnostics = {
        "experiment_name": str(cfg.experiment_name),
        "codebook_size": int(cfg.model.codebook_size),
        "chance_token_acc": float(1.0 / int(cfg.model.codebook_size)),
        "token_entropy": float(torch.cat(entropy_values).mean().item()),
        "codebook_perplexity": float(torch.exp(codebook_entropy).item()),
        "top1_token_ratio": float((histogram.max() / histogram.sum().clamp_min(1.0)).item()),
        "unique_token_count": int((histogram > 0).sum().item()),
        "token_acc_normal_conditioning": float(np.mean(normal_acc)),
        "token_acc_shuffled_future_targets": float(np.mean(shuffled_acc)),
        "token_acc_zeroed_current_observation": float(np.mean(zero_obs_acc)),
    }

    output_path = Path(cfg.output_root) / cfg.experiment_name / "planner_diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
