from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.common import resolve_h5_path
from ..subgoal_system import VW2SubgoalDataModule, VW2SubgoalSystem
from ..system import VW2DirectActSystem
from ..utils.metrics import batch_action_mse
from ..utils.config import load_config
from ..utils.rollout import SubgoalPolicy
from .eval_policy import (
    _normalize_sequence_pixels,
    _offline_metrics as _legacy_offline_metrics,
    _resolve_eval_max_steps,
    _resize_hwc_uint8,
    _save_rollout_videos,
    _world_metrics as _legacy_world_metrics,
)

DEFAULT_EXECUTE_SWEEP = (1, 2)

os.environ.setdefault("WANDB_CONSOLE", "off")
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BC and subgoal-distillation policies.")
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--student-frozen-checkpoint", default=None)
    parser.add_argument("--student-joint-checkpoint", default=None)
    return parser


def _resolve_execute_sweep(cfg) -> list[int]:
    values = cfg.eval.get("subgoal_execute_actions_per_plan_sweep", None)
    if values is None:
        return list(DEFAULT_EXECUTE_SWEEP)
    return [int(value) for value in values]


def _resolve_world_dataset(cfg):
    import stable_worldmodel as swm

    resolved = resolve_h5_path(cfg.data.path, cfg.data.dataset_name, cfg.data.cache_dir)
    return swm.data.HDF5Dataset(resolved.stem, cache_dir=str(resolved.parent))


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _load_subgoal_system(cfg, checkpoint: str) -> VW2SubgoalSystem:
    system = VW2SubgoalSystem(cfg, "joint_subgoal")
    system.load_weights_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system.to(device)
    system.eval()
    return system


def _subgoal_offline_metrics(system: VW2SubgoalSystem, cfg, *, mode: str) -> dict[str, float]:
    datamodule = VW2SubgoalDataModule(cfg)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    device = next(system.parameters()).device
    action_mse_values = []
    diagnostics = {
        "subgoal_batch_variance": [],
        "covariance_offdiag_mean": [],
        "retrieval_top1": [],
        "retrieval_top1_shuffled": [],
        "retrieval_chance": [],
        "action_gap": [],
    }

    for batch_index, batch in enumerate(loader):
        if batch_index >= int(cfg.eval.offline_batches):
            break
        batch = _to_device(batch, device)
        with torch.no_grad():
            context = system.model.encode_history(
                history_pixels=batch["history_pixels"],
                history_proprio=batch.get("history_proprio"),
                prev_actions=batch["prev_actions"],
            )
            target = batch["action"][:, : int(cfg.model.action_chunk)]
            teacher_subgoal = system.model.encode_future_subgoal(
                future_pixels=batch["future_pixels"],
                future_proprio=batch.get("future_proprio"),
            )
            if mode == "bc":
                subgoal = system.model.zero_subgoal(context.shape[0], device=device)
                predicted = system.model.act(context=context, subgoal=subgoal)
            elif mode == "oracle":
                predicted = system.model.act(context=context, subgoal=teacher_subgoal)
            elif mode == "student":
                predicted_subgoal = system.model.predict_subgoal(
                    context=context,
                    horizon_steps=batch["horizon"],
                )
                predicted = system.model.act(context=context, subgoal=predicted_subgoal)
                teacher_actions = system.model.act(context=context, subgoal=teacher_subgoal)
                diagnostics["subgoal_batch_variance"].append(float(predicted_subgoal.var(dim=0, unbiased=False).mean().cpu()))
                centered = predicted_subgoal - predicted_subgoal.mean(dim=0, keepdim=True)
                covariance = centered.transpose(0, 1) @ centered
                covariance = covariance / max(predicted_subgoal.shape[0] - 1, 1)
                mask = ~torch.eye(covariance.shape[0], device=device, dtype=torch.bool)
                offdiag = covariance.masked_select(mask)
                diagnostics["covariance_offdiag_mean"].append(float(offdiag.abs().mean().cpu()) if offdiag.numel() > 0 else 0.0)
                similarity = torch.nn.functional.normalize(predicted_subgoal, dim=-1) @ torch.nn.functional.normalize(teacher_subgoal, dim=-1).transpose(0, 1)
                labels = torch.arange(predicted_subgoal.shape[0], device=device)
                diagnostics["retrieval_top1"].append(float((similarity.argmax(dim=1) == labels).float().mean().cpu()))
                permutation = torch.roll(labels, shifts=1)
                shuffled_similarity = torch.nn.functional.normalize(predicted_subgoal, dim=-1) @ torch.nn.functional.normalize(teacher_subgoal[permutation], dim=-1).transpose(0, 1)
                diagnostics["retrieval_top1_shuffled"].append(float((shuffled_similarity.argmax(dim=1) == labels).float().mean().cpu()))
                diagnostics["retrieval_chance"].append(float(1.0 / predicted_subgoal.shape[0]))
                diagnostics["action_gap"].append(float(batch_action_mse(predicted, teacher_actions).cpu()))
            else:
                raise ValueError(f"Unsupported offline mode={mode!r}.")
            action_mse_values.append(float(batch_action_mse(predicted, target).cpu()))

    metrics = {
        "action_mse": float(np.mean(action_mse_values)) if action_mse_values else float("nan"),
    }
    for key, values in diagnostics.items():
        if values:
            metrics[key] = float(np.mean(values))
    datamodule.teardown("eval")
    return metrics


def _select_eval_starts(dataset, cfg) -> tuple[np.ndarray, np.ndarray]:
    episode_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(episode_key)
    step_idx = dataset.get_col_data("step_idx")
    episodes = np.unique(episode_ids)
    required_future = max(
        int(cfg.eval.goal_offset_steps),
        _resolve_eval_max_steps(cfg) + int(cfg.data.plan_horizon),
    )
    min_start = int(cfg.subgoal.history_steps) - 1
    starts = []
    for episode in episodes:
        mask = episode_ids == episode
        episode_steps = step_idx[mask]
        limit = int(episode_steps.max()) - required_future
        valid = np.where(mask & (step_idx >= min_start) & (step_idx <= limit))[0]
        if valid.size > 0:
            starts.append(valid[0])
    chosen = np.array(starts[: int(cfg.eval.num_rollouts)], dtype=np.int64)
    rows = dataset.get_row_data(chosen.tolist())
    return np.asarray(rows[episode_key]), np.asarray(rows["step_idx"])


def _build_subgoal_rollout_state(
    world,
    dataset,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    cfg,
    *,
    max_steps: int,
    world_image_shape: tuple[int, int],
) -> dict[str, Any]:
    history_steps = int(cfg.subgoal.history_steps)
    goal_offset_steps = int(cfg.eval.goal_offset_steps)
    chunk_length = max(goal_offset_steps, max_steps + int(cfg.data.plan_horizon))
    load_start = start_steps - (history_steps - 1)
    load_end = start_steps + chunk_length
    data = dataset.load_chunk(episodes_idx, load_start, load_end)
    columns = dataset.column_names
    current_offset = history_steps - 1

    init_step_per_env: dict[str, list[Any]] = {}
    goal_step_per_env: dict[str, list[Any]] = {}
    for episode in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            value = episode[col]
            if col.startswith("pixels"):
                value = value.permute(0, 2, 3, 1)
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            init_data = value[current_offset]
            goal_data = value[min(current_offset + goal_offset_steps - 1, value.shape[0] - 1)]
            if isinstance(init_data, torch.Tensor):
                init_data = init_data.numpy()
            if isinstance(goal_data, torch.Tensor):
                goal_data = goal_data.numpy()
            if col.startswith("pixels"):
                init_data = _resize_hwc_uint8(init_data[None], world_image_shape)[0]
                goal_data = _resize_hwc_uint8(goal_data[None], world_image_shape)[0]
            init_step_per_env.setdefault(col, []).append(init_data)
            goal_step_per_env.setdefault(col, []).append(goal_data)

    init_step = {key: np.stack(values) for key, values in init_step_per_env.items()}
    goal_step = {}
    for key, values in goal_step_per_env.items():
        mapped_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[mapped_key] = np.stack(values)

    seeds = init_step.get("seed")
    variations = {
        key.removeprefix("variation."): value
        for key, value in init_step.items()
        if key.startswith("variation.")
    }

    options = [{} for _ in range(world.num_envs)]
    if variations:
        for index in range(world.num_envs):
            options[index]["variation"] = list(variations.keys())
            options[index]["variation_values"] = {key: value[index] for key, value in variations.items()}

    init_with_goal = init_step | goal_step
    world.reset(seed=seeds, options=options)

    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ]
    for env_index, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            if not hasattr(env_unwrapped, spec["method"]):
                continue
            method = getattr(env_unwrapped, spec["method"])
            prepared_args = {}
            for arg_name, arg_data in spec["args"].items():
                column = arg_data.get("value")
                if column not in init_with_goal:
                    continue
                prepared_args[arg_name] = deepcopy(init_with_goal[column][env_index])
            method(**prepared_args)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        key: np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:])
        for key, value in init_with_goal.items()
    }
    goal_step = {
        key: np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:])
        for key, value in goal_step.items()
    }
    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(goal_step))

    bootstrap_history = []
    bootstrap_proprio = []
    bootstrap_prev_actions = []
    oracle_rollout_pixels = []
    oracle_rollout_proprio = []
    target_frames = []
    for episode in data:
        history_pixels = episode["pixels"][:history_steps]
        actions = episode["action"][: history_steps - 1]
        prev_actions = torch.zeros(history_steps, int(cfg.model.action_dim), dtype=torch.float32)
        if history_steps > 1:
            prev_actions[1:] = actions.float()
        bootstrap_history.append(history_pixels.float())
        oracle_rollout_pixels.append(episode["pixels"].float())
        bootstrap_prev_actions.append(prev_actions)
        if "proprio" in episode:
            bootstrap_proprio.append(episode["proprio"][:history_steps].float())
            oracle_rollout_proprio.append(episode["proprio"].float())
        frames = episode["pixels"][current_offset : current_offset + max_steps + 1].permute(0, 2, 3, 1).numpy()
        target_frames.append(_resize_hwc_uint8(frames, world_image_shape))

    return {
        "goal_step": goal_step,
        "target_frames": np.stack(target_frames),
        "bootstrap_history_pixels": torch.stack(bootstrap_history, dim=0),
        "bootstrap_history_proprio": None if not bootstrap_proprio else torch.stack(bootstrap_proprio, dim=0),
        "bootstrap_prev_actions": torch.stack(bootstrap_prev_actions, dim=0),
        "oracle_rollout_pixels": torch.stack(oracle_rollout_pixels, dim=0),
        "oracle_rollout_proprio": None if not oracle_rollout_proprio else torch.stack(oracle_rollout_proprio, dim=0),
        "episodes_idx": episodes_idx.tolist(),
        "start_steps": start_steps.tolist(),
        "seeds": None if seeds is None else np.asarray(seeds).tolist(),
    }


def _oracle_subgoals_from_rollout_sequence(
    system: VW2SubgoalSystem,
    setup: dict[str, Any],
    cfg,
    *,
    device: torch.device,
    max_steps: int,
) -> torch.Tensor:
    history_steps = int(cfg.subgoal.history_steps)
    future_horizon = int(cfg.data.plan_horizon)
    step_batch = 8
    rollout_pixels = setup["oracle_rollout_pixels"]
    rollout_proprio = setup["oracle_rollout_proprio"]
    outputs = []
    with torch.no_grad():
        for start_step in range(0, max_steps, step_batch):
            end_step = min(start_step + step_batch, max_steps)
            pixel_windows = torch.stack(
                [
                    torch.stack(
                        [episode[history_steps + step : history_steps + step + future_horizon] for episode in rollout_pixels],
                        dim=0,
                    )
                    for step in range(start_step, end_step)
                ],
                dim=1,
            )
            batch_size, num_steps = pixel_windows.shape[:2]
            normalized_pixels = _normalize_sequence_pixels(
                pixel_windows,
                image_size=int(cfg.data.image_size),
                device=device,
            )
            future_pixels = normalized_pixels.reshape(
                batch_size * num_steps,
                future_horizon,
                *normalized_pixels.shape[-3:],
            )
            future_proprio = None
            if rollout_proprio is not None:
                proprio_windows = torch.stack(
                    [
                        torch.stack(
                            [episode[history_steps + step : history_steps + step + future_horizon] for episode in rollout_proprio],
                            dim=0,
                        )
                        for step in range(start_step, end_step)
                    ],
                    dim=1,
                ).float()
                future_proprio = proprio_windows.to(device).reshape(
                    batch_size * num_steps,
                    future_horizon,
                    proprio_windows.shape[-1],
                )
            subgoals = system.model.encode_future_subgoal(
                future_pixels=future_pixels,
                future_proprio=future_proprio,
            )
            outputs.append(subgoals.reshape(batch_size, num_steps, -1).cpu())
    return torch.cat(outputs, dim=1)


def _run_subgoal_world_batch(
    system: VW2SubgoalSystem,
    cfg,
    *,
    dataset,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    execute_steps: int,
    mode: str,
    save_video_count: int,
    video_dir: Path,
    video_offset: int,
) -> dict[str, Any]:
    import stable_worldmodel as swm

    world_image_shape = (int(cfg.data.image_size), int(cfg.data.image_size))
    requested_max_steps = cfg.eval.get("max_steps", None)
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=len(episodes_idx),
        history_size=1,
        frame_skip=1,
        max_episode_steps=None if requested_max_steps is None else int(requested_max_steps),
        image_shape=world_image_shape,
        verbose=0,
    )
    try:
        max_steps = world.envs.envs[0].spec.max_episode_steps
        setup = _build_subgoal_rollout_state(
            world,
            dataset,
            episodes_idx,
            start_steps,
            cfg,
            max_steps=max_steps,
            world_image_shape=world_image_shape,
        )
        device = next(system.parameters()).device
        bootstrap_history_pixels = _normalize_sequence_pixels(
            setup["bootstrap_history_pixels"],
            image_size=int(cfg.data.image_size),
            device=device,
        )
        bootstrap_history_proprio = None
        if setup["bootstrap_history_proprio"] is not None:
            bootstrap_history_proprio = setup["bootstrap_history_proprio"].float().to(device)
        bootstrap_prev_actions = setup["bootstrap_prev_actions"].float().to(device)

        oracle_subgoal = None
        oracle_subgoals_by_step = None
        if mode == "oracle":
            oracle_subgoals_by_step = _oracle_subgoals_from_rollout_sequence(
                system,
                setup,
                cfg,
                device=device,
                max_steps=max_steps,
            )

        policy = SubgoalPolicy(
            model=system.model,
            image_size=int(cfg.data.image_size),
            history_steps=int(cfg.subgoal.history_steps),
            action_dim=int(cfg.model.action_dim),
            execute_steps=execute_steps,
            horizon_steps=int(cfg.data.plan_horizon),
            mode=mode,
            oracle_subgoal=oracle_subgoal,
            oracle_subgoals_by_step=oracle_subgoals_by_step,
            bootstrap_history_pixels=bootstrap_history_pixels,
            bootstrap_history_proprio=bootstrap_history_proprio,
            bootstrap_prev_actions=bootstrap_prev_actions,
        )
        world.set_policy(policy)

        reward_traces = np.zeros((world.num_envs, max_steps), dtype=np.float32)
        success_traces = np.zeros((world.num_envs, max_steps), dtype=np.bool_)
        capture_count = min(save_video_count, world.num_envs)
        video_frames = None
        if capture_count > 0:
            video_frames = np.empty(
                (capture_count, max_steps + 1, *world.infos["pixels"].shape[-3:]),
                dtype=np.uint8,
            )
            video_frames[:, 0] = world.infos["pixels"][:capture_count, -1]

        cumulative_success = np.zeros(world.num_envs, dtype=np.bool_)
        for step in range(max_steps):
            world.infos.update(deepcopy(setup["goal_step"]))
            world.step()
            reward_traces[:, step] = np.asarray(world.rewards, dtype=np.float32)
            cumulative_success = np.logical_or(cumulative_success, np.asarray(world.terminateds, dtype=np.bool_))
            success_traces[:, step] = cumulative_success
            if video_frames is not None:
                video_frames[:, step + 1] = world.infos["pixels"][:capture_count, -1]
            world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,), dtype=np.bool_)

        saved_paths: list[str] = []
        if video_frames is not None:
            saved_paths = _save_rollout_videos(
                video_frames=video_frames,
                target_frames=setup["target_frames"][:capture_count],
                output_dir=video_dir,
                global_offset=video_offset,
            )

        episode_rewards = reward_traces.sum(axis=1)
        return {
            "max_steps": int(max_steps),
            "episode_successes": cumulative_success.tolist(),
            "reward_traces": reward_traces.tolist(),
            "success_traces": success_traces.tolist(),
            "episode_rewards": episode_rewards.tolist(),
            "mean_episode_reward": float(episode_rewards.mean()),
            "video_paths": saved_paths,
            "episodes_idx": setup["episodes_idx"],
            "start_steps": setup["start_steps"],
        }
    finally:
        world.close()


def _subgoal_world_metrics(system: VW2SubgoalSystem, cfg, *, mode: str, execute_steps: int, label: str) -> dict[str, Any]:
    if str(cfg.data.dataset_type) != "pusht" or not bool(cfg.eval.run_world):
        return {}
    try:
        _resolve_world_dataset(cfg)
    except Exception:
        return {}

    dataset = _resolve_world_dataset(cfg)
    eval_episodes, eval_starts = _select_eval_starts(dataset, cfg)
    if eval_episodes.size == 0:
        return {}

    rollout_batch_size = int(cfg.eval.rollout_batch_size)
    output_dir = Path(cfg.output_root) / cfg.experiment_name / f"eval_subgoal_{int(cfg.eval.num_rollouts)}rollouts_{int(cfg.eval.max_steps)}steps" / label
    video_dir = output_dir / f"videos_execute_{execute_steps}"
    videos_remaining = int(cfg.eval.save_video_count) if bool(cfg.eval.save_video) and execute_steps == _resolve_execute_sweep(cfg)[0] else 0
    video_offset = 0

    episode_successes: list[bool] = []
    reward_traces: list[list[float]] = []
    success_traces: list[list[bool]] = []
    episode_rewards: list[float] = []
    video_paths: list[str] = []
    rollout_episode_ids: list[int] = []
    rollout_start_steps: list[int] = []

    for start in range(0, len(eval_episodes), rollout_batch_size):
        batch_slice = slice(start, min(start + rollout_batch_size, len(eval_episodes)))
        batch = _run_subgoal_world_batch(
            system,
            cfg,
            dataset=dataset,
            episodes_idx=eval_episodes[batch_slice],
            start_steps=eval_starts[batch_slice],
            execute_steps=execute_steps,
            mode=mode,
            save_video_count=videos_remaining,
            video_dir=video_dir,
            video_offset=video_offset,
        )
        episode_successes.extend(bool(value) for value in batch["episode_successes"])
        reward_traces.extend(batch["reward_traces"])
        success_traces.extend(batch["success_traces"])
        episode_rewards.extend(float(value) for value in batch["episode_rewards"])
        video_paths.extend(batch["video_paths"])
        rollout_episode_ids.extend(int(value) for value in batch["episodes_idx"])
        rollout_start_steps.extend(int(value) for value in batch["start_steps"])
        videos_remaining = max(0, videos_remaining - len(batch["video_paths"]))
        video_offset += len(batch["video_paths"])

    success_rate = float(np.mean(np.asarray(episode_successes, dtype=np.float32)) * 100.0)
    return {
        "num_rollouts": len(episode_successes),
        "max_steps": int(cfg.eval.max_steps),
        "execute_actions_per_plan": execute_steps,
        "success_rate": success_rate,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        "episode_successes": episode_successes,
        "reward_traces": reward_traces,
        "success_traces": success_traces,
        "episode_rewards": episode_rewards,
        "video_paths": video_paths,
        "episode_ids": rollout_episode_ids,
        "start_steps": rollout_start_steps,
    }


def _write_per_episode_csv(path: Path, label: str, execute_steps: int, world_metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_paths = list(world_metrics.get("video_paths", []))
    episode_successes = list(world_metrics.get("episode_successes", []))
    episode_ids = list(world_metrics.get("episode_ids", [None] * len(episode_successes)))
    start_steps = list(world_metrics.get("start_steps", [None] * len(episode_successes)))
    episode_rewards = list(world_metrics.get("episode_rewards", [float("nan")] * len(episode_successes)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "execute_actions_per_plan",
                "rollout_index",
                "episode_id",
                "start_step",
                "success",
                "episode_reward",
                "video_path",
            ],
        )
        writer.writeheader()
        for index, success in enumerate(episode_successes):
            writer.writerow(
                {
                    "mode": label,
                    "execute_actions_per_plan": execute_steps,
                    "rollout_index": index,
                    "episode_id": episode_ids[index],
                    "start_step": start_steps[index],
                    "success": int(bool(success)),
                    "episode_reward": episode_rewards[index],
                    "video_path": video_paths[index] if index < len(video_paths) else "",
                }
            )


def _evaluate_legacy_bc(cfg, checkpoint: str, execute_sweep: list[int], output_dir: Path) -> dict[str, Any]:
    system = VW2DirectActSystem(cfg, "joint")
    system.load_weights_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system.to(device)
    system.eval()
    offline = _legacy_offline_metrics(system, cfg, conditioning_mode="bc")
    world_results = {}
    for execute_steps in execute_sweep:
        world = _legacy_world_metrics(
            system,
            cfg,
            conditioning_mode="bc",
            execute_steps=execute_steps,
            save_videos=bool(cfg.eval.save_video) and execute_steps == execute_sweep[0],
        )
        world_results[str(execute_steps)] = world
        _write_per_episode_csv(output_dir / "BC" / f"execute_{execute_steps}.csv", "BC", execute_steps, world)
    return {"offline": offline, "world": world_results}


def _evaluate_subgoal(cfg, checkpoint: str, *, label: str, mode: str, execute_sweep: list[int], output_dir: Path) -> dict[str, Any]:
    system = _load_subgoal_system(cfg, checkpoint)
    offline = _subgoal_offline_metrics(system, cfg, mode=mode)
    world_results = {}
    for execute_steps in execute_sweep:
        world = _subgoal_world_metrics(system, cfg, mode=mode, execute_steps=execute_steps, label=label)
        world_results[str(execute_steps)] = world
        _write_per_episode_csv(output_dir / label / f"execute_{execute_steps}.csv", label, execute_steps, world)
    return {"offline": offline, "world": world_results}


def _teacher_gate(results: dict[str, Any]) -> bool:
    execute_1 = results["world"].get("1", {}).get("success_rate", float("-inf"))
    execute_2 = results["world"].get("2", {}).get("success_rate", float("-inf"))
    return execute_1 >= 90.0 and execute_2 >= 90.0


def _student_success_gate(results: dict[str, Any]) -> bool:
    execute_1 = results["world"].get("1", {}).get("success_rate", float("-inf"))
    execute_2 = results["world"].get("2", {}).get("success_rate", float("-inf"))
    return execute_1 >= 50.0 and execute_2 >= 30.0


def _student_vs_bc_gate(student_results: dict[str, Any], bc_results: dict[str, Any]) -> bool:
    for execute in ("1", "2"):
        student_success = student_results["world"].get(execute, {}).get("success_rate", float("-inf"))
        bc_success = bc_results["world"].get(execute, {}).get("success_rate", float("inf"))
        if student_success < bc_success + 5.0:
            return False
    return True


def _retrieval_gate(results: dict[str, Any]) -> bool:
    shuffled = results["offline"].get("retrieval_top1_shuffled")
    chance = results["offline"].get("retrieval_chance")
    if shuffled is None or chance is None:
        return False
    return shuffled <= chance + 0.05


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = _parser()
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config_name, config_path=args.config_path, overrides=overrides)

    execute_sweep = _resolve_execute_sweep(cfg)
    output_dir = Path(cfg.output_root) / cfg.experiment_name / f"eval_subgoal_{int(cfg.eval.num_rollouts)}rollouts_{int(cfg.eval.max_steps)}steps"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    if args.bc_checkpoint:
        results["BC"] = _evaluate_legacy_bc(cfg, args.bc_checkpoint, execute_sweep, output_dir)
        _write_summary(output_dir, results)

    if not args.teacher_checkpoint:
        raise ValueError("--teacher-checkpoint is required for subgoal evaluation.")
    results["TeacherOracle"] = _evaluate_subgoal(
        cfg,
        args.teacher_checkpoint,
        label="TeacherOracle",
        mode="oracle",
        execute_sweep=execute_sweep,
        output_dir=output_dir,
    )
    gates = {
        "GateA_TeacherOracle": {
            "pass": _teacher_gate(results["TeacherOracle"]),
            "rule": "TeacherOracle success >= 90% on execute-1 and execute-2.",
        }
    }
    _write_summary(output_dir, {"results": results, "gates": gates})
    if not gates["GateA_TeacherOracle"]["pass"]:
        gates["hard_stop"] = {
            "stop": True,
            "reason": "Gate A failed. Stop the future-conditioned branch immediately.",
        }
        _write_summary(output_dir, {"results": results, "gates": gates})
        print(json.dumps({"results": results, "gates": gates}, indent=2))
        return

    if args.student_frozen_checkpoint:
        results["StudentFrozen"] = _evaluate_subgoal(
            cfg,
            args.student_frozen_checkpoint,
            label="StudentFrozen",
            mode="student",
            execute_sweep=execute_sweep,
            output_dir=output_dir,
        )
        _write_summary(output_dir, {"results": results, "gates": gates})

    if not args.student_joint_checkpoint:
        raise ValueError("--student-joint-checkpoint is required for full gate evaluation.")
    results["StudentJoint"] = _evaluate_subgoal(
        cfg,
        args.student_joint_checkpoint,
        label="StudentJoint",
        mode="student",
        execute_sweep=execute_sweep,
        output_dir=output_dir,
    )

    retrieval_reference = results["StudentJoint"]
    gates["GateB_ShuffledRetrieval"] = {
        "pass": _retrieval_gate(retrieval_reference),
        "rule": "Student shuffled future retrieval must collapse to near chance (chance + 5 percentage points).",
        "value": retrieval_reference["offline"].get("retrieval_top1_shuffled"),
        "chance": retrieval_reference["offline"].get("retrieval_chance"),
    }
    gates["GateC_StudentJoint"] = {
        "pass": _student_success_gate(results["StudentJoint"]),
        "rule": "StudentJoint success >= 50% on execute-1 and >= 30% on execute-2.",
    }
    if "BC" in results:
        gates["GateD_StudentBeatsBC"] = {
            "pass": _student_vs_bc_gate(results["StudentJoint"], results["BC"]),
            "rule": "StudentJoint success exceeds BC by at least 5 percentage points on execute-1 and execute-2.",
        }
    else:
        gates["GateD_StudentBeatsBC"] = {
            "pass": False,
            "rule": "BC checkpoint missing, so StudentJoint vs BC cannot be evaluated.",
        }
    gates["hard_stop"] = {
        "stop": not gates["GateC_StudentJoint"]["pass"],
        "reason": "Gate C failed after one joint run. Stop the whole future-conditioned branch on Push-T."
        if not gates["GateC_StudentJoint"]["pass"]
        else "",
    }

    summary = {"results": results, "gates": gates}
    _write_summary(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
