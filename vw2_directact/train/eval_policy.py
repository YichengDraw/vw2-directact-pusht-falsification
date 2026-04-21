from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.common import IMAGENET_MEAN, IMAGENET_STD, resolve_h5_path
from ..system import VW2DirectActDataModule, VW2DirectActSystem
from ..utils.metrics import batch_action_mse, latency_ms
from ..utils.rollout import DirectActPolicy
from .common import load_cfg_for_eval

DEFAULT_EXECUTE_SWEEP = (1, 2, 4)
DEFAULT_ROLLOUT_BATCH_SIZE = 10
DEFAULT_VIDEO_COUNT = 10


def _resolve_world_image_shape(dataset, fallback_size: int) -> tuple[int, int]:
    sample = dataset.get_row_data(np.array([0]))
    pixels = sample.get("pixels")
    if pixels is None or pixels.ndim < 4:
        return (fallback_size, fallback_size)
    height, width = pixels.shape[-3:-1]
    return int(height), int(width)


def _resize_hwc_uint8(images: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=image_shape, mode="bilinear", align_corners=False)
    tensor = tensor.clamp(0.0, 255.0).round().byte().permute(0, 2, 3, 1).contiguous()
    return tensor.numpy()


def _resolve_conditioning_mode(cfg) -> str:
    conditioning = cfg.get("conditioning")
    if conditioning is not None and conditioning.get("mode") is not None:
        return str(conditioning.mode)
    if str(cfg.ablation.mode) == "bc":
        return "bc"
    return "predfuture"


def _resolve_execute_sweep(cfg) -> list[int]:
    values = cfg.eval.get("execute_actions_per_plan_sweep", None)
    if values is None:
        return list(DEFAULT_EXECUTE_SWEEP)
    return [int(value) for value in values]


def _resolve_rollout_batch_size(cfg) -> int:
    return int(cfg.eval.get("rollout_batch_size", DEFAULT_ROLLOUT_BATCH_SIZE))


def _resolve_video_count(cfg) -> int:
    return int(cfg.eval.get("save_video_count", DEFAULT_VIDEO_COUNT))


def _resolve_eval_max_steps(cfg, *, fallback: int | None = None) -> int:
    value = cfg.eval.get("max_steps", None)
    if value is not None:
        return int(value)
    if fallback is not None:
        return int(fallback)
    raise ValueError("eval.max_steps is unset and no fallback max_episode_steps was provided.")


def _resolve_world_dataset(cfg):
    import stable_worldmodel as swm

    resolved = resolve_h5_path(cfg.data.path, cfg.data.dataset_name, cfg.data.cache_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Push-T HDF5 dataset not found: {resolved}")
    return swm.data.HDF5Dataset(resolved.stem, cache_dir=str(resolved.parent))


def _to_device(batch, device):
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def _normalize_sequence_pixels(pixels: torch.Tensor, *, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = pixels.float().to(device)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    original_shape = tensor.shape[:-3]
    tensor = tensor.reshape(-1, *tensor.shape[-3:])
    if tensor.shape[-1] != image_size or tensor.shape[-2] != image_size:
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(device=device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(device=device, dtype=tensor.dtype)
    tensor = (tensor - mean) / std
    return tensor.reshape(*original_shape, *tensor.shape[-3:])


def _oracle_plan_embeddings_from_chunk(
    system: VW2DirectActSystem,
    data: list[dict[str, torch.Tensor]],
    cfg,
    *,
    device: torch.device,
    max_steps: int,
) -> torch.Tensor:
    sequence_length = int(cfg.data.plan_horizon) + 1
    step_batch = 8
    outputs = []
    with torch.no_grad():
        for start_step in range(0, max_steps, step_batch):
            end_step = min(start_step + step_batch, max_steps)
            pixels = torch.stack(
                [
                    torch.stack([episode["pixels"][step : step + sequence_length] for episode in data], dim=0)
                    for step in range(start_step, end_step)
                ],
                dim=1,
            )
            batch_size, num_steps = pixels.shape[:2]
            normalized_pixels = _normalize_sequence_pixels(
                pixels,
                image_size=int(cfg.data.image_size),
                device=device,
            )
            batch: dict[str, torch.Tensor] = {
                "pixels": normalized_pixels.reshape(
                    batch_size * num_steps,
                    sequence_length,
                    *normalized_pixels.shape[-3:],
                )
            }
            if "proprio" in data[0]:
                proprio = torch.stack(
                    [
                        torch.stack([episode["proprio"][step : step + sequence_length] for episode in data], dim=0)
                        for step in range(start_step, end_step)
                    ],
                    dim=1,
                ).float()
                batch["proprio"] = proprio.to(device).reshape(batch_size * num_steps, sequence_length, proprio.shape[-1])
            teacher_plan = system._teacher_plan(batch, detach=True)
            outputs.append(
                teacher_plan["plan_embeddings"].reshape(
                    batch_size,
                    num_steps,
                    teacher_plan["plan_embeddings"].shape[1],
                    teacher_plan["plan_embeddings"].shape[2],
                ).cpu()
            )
    return torch.cat(outputs, dim=1)


def _offline_metrics(system: VW2DirectActSystem, cfg, *, conditioning_mode: str) -> dict[str, float]:
    datamodule = VW2DirectActDataModule(cfg, "joint")
    datamodule.setup()
    loader = datamodule.val_dataloader()
    device = next(system.parameters()).device
    mse_values = []
    token_acc_values = []
    latency_values = []

    for batch_index, batch in enumerate(loader):
        if batch_index >= int(cfg.eval.offline_batches):
            break
        batch = _to_device(batch, device)
        pixels = batch["pixels"][:, 0]
        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio[:, 0]
        gripper = batch.get("gripper_pixels")
        if gripper is not None:
            gripper = gripper[:, 0]
        language = batch.get("language")
        if language is not None:
            language = language[:, 0]

        plan_override = None
        if conditioning_mode == "oracle":
            teacher_plan = system._teacher_plan(batch, detach=True)
            plan_override = teacher_plan["plan_embeddings"]
        else:
            teacher_plan = None

        latency_values.append(
            latency_ms(
                lambda: system.model.predict_action_chunk(
                    pixels=pixels,
                    proprio=proprio,
                    gripper_pixels=gripper,
                    language=language,
                    temperature=float(cfg.sampling.temperature),
                    mode=conditioning_mode,
                    plan_override=plan_override,
                )
            )
        )
        with torch.no_grad():
            predicted = system.model.predict_action_chunk(
                pixels=pixels,
                proprio=proprio,
                gripper_pixels=gripper,
                language=language,
                temperature=float(cfg.sampling.temperature),
                mode=conditioning_mode,
                plan_override=plan_override,
            )
            target = batch["action"][:, : int(cfg.model.action_chunk)]
            mse_values.append(float(batch_action_mse(predicted, target).cpu()))
            if bool(cfg.model.use_vq) and conditioning_mode == "predfuture":
                current = system._encode_current(batch)
                teacher_plan = teacher_plan or system._teacher_plan(batch, detach=True)
                _, token_acc = system._plan_loss(current, teacher_plan)
                token_acc_values.append(float(token_acc.cpu()))

    metrics = {
        "action_mse": float(np.mean(mse_values)) if mse_values else float("nan"),
        "latency_ms": float(np.mean(latency_values)) if latency_values else float("nan"),
    }
    if token_acc_values:
        metrics["token_accuracy"] = float(np.mean(token_acc_values))
    return metrics


def _select_eval_starts(dataset, cfg) -> tuple[np.ndarray, np.ndarray]:
    episode_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(episode_key)
    step_idx = dataset.get_col_data("step_idx")
    episodes = np.unique(episode_ids)
    required_length = max(int(cfg.eval.goal_offset_steps), _resolve_eval_max_steps(cfg) + int(cfg.data.plan_horizon))
    lengths = np.array([step_idx[episode_ids == episode].max() + 1 for episode in episodes])
    max_start = lengths - required_length
    starts = []
    for episode, limit in zip(episodes, max_start):
        if limit < 0:
            continue
        valid = np.where((episode_ids == episode) & (step_idx <= limit))[0]
        if valid.size > 0:
            starts.append(valid[0])
    chosen = np.array(starts[: int(cfg.eval.num_rollouts)], dtype=np.int64)
    rows = dataset.get_row_data(chosen.tolist())
    return np.asarray(rows[episode_key]), np.asarray(rows["step_idx"])


def _build_rollout_state(
    world,
    dataset,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    cfg,
    *,
    world_image_shape: tuple[int, int],
) -> dict[str, Any]:
    goal_offset_steps = int(cfg.eval.goal_offset_steps)
    chunk_length = max(goal_offset_steps, _resolve_eval_max_steps(cfg) + int(cfg.data.plan_horizon))
    end_steps = start_steps + chunk_length
    data = dataset.load_chunk(episodes_idx, start_steps, end_steps)
    columns = dataset.column_names

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
            init_data = value[0]
            goal_data = value[min(goal_offset_steps - 1, value.shape[0] - 1)]
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

    target_frames = []
    for episode in data:
        frames = episode["pixels"][: _resolve_eval_max_steps(cfg) + 1].permute(0, 2, 3, 1).numpy()
        target_frames.append(_resize_hwc_uint8(frames, world_image_shape))

    return {
        "data": data,
        "goal_step": goal_step,
        "target_frames": np.stack(target_frames),
        "seeds": None if seeds is None else np.asarray(seeds).tolist(),
    }


def _save_rollout_videos(
    *,
    video_frames: np.ndarray,
    target_frames: np.ndarray,
    output_dir: Path,
    global_offset: int,
) -> list[str]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    target_len = target_frames.shape[1]
    saved_paths = []
    for index in range(video_frames.shape[0]):
        path = output_dir / f"rollout_{global_offset + index:02d}.mp4"
        first_demo = target_frames[index, 0]
        first_frame = np.hstack(
            [
                np.vstack([video_frames[index, 0], first_demo]),
                np.vstack([target_frames[index, -1], target_frames[index, -1]]),
            ]
        )
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            15.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {path}")
        goal_panel = np.vstack([target_frames[index, -1], target_frames[index, -1]])
        for step in range(video_frames.shape[1] - 1):
            demo_frame = target_frames[index, min(step, target_len - 1)]
            stacked = np.vstack([video_frames[index, step], demo_frame])
            frame = np.hstack([stacked, goal_panel])
            writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        saved_paths.append(str(path))
    return saved_paths


def _run_world_batch(
    system: VW2DirectActSystem,
    cfg,
    *,
    dataset,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    execute_steps: int,
    conditioning_mode: str,
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
        if max_steps is None:
            raise ValueError("max_steps is unset. Provide eval.max_steps or an environment max_episode_steps.")

        setup = _build_rollout_state(
            world,
            dataset,
            episodes_idx,
            start_steps,
            cfg,
            world_image_shape=world_image_shape,
        )
        oracle_plan_embeddings = None
        if conditioning_mode == "oracle":
            oracle_plan_embeddings = _oracle_plan_embeddings_from_chunk(
                system,
                setup["data"],
                cfg,
                device=next(system.parameters()).device,
                max_steps=max_steps,
            )
        policy = DirectActPolicy(
            model=system.model,
            image_size=int(cfg.data.image_size),
            execute_steps=execute_steps,
            mode=conditioning_mode,
            temperature=float(cfg.sampling.temperature),
            oracle_plan_embeddings_by_step=oracle_plan_embeddings,
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
            "seeds": setup["seeds"],
        }
    finally:
        world.close()


def _world_metrics(system: VW2DirectActSystem, cfg, *, conditioning_mode: str, execute_steps: int, save_videos: bool) -> dict[str, Any]:
    if str(cfg.data.dataset_type) != "pusht" or not bool(cfg.eval.run_world):
        return {}

    dataset = _resolve_world_dataset(cfg)
    eval_episodes, eval_starts = _select_eval_starts(dataset, cfg)
    if eval_episodes.size == 0:
        raise ValueError("No valid Push-T world-evaluation rollout starts were found.")

    rollout_batch_size = max(1, _resolve_rollout_batch_size(cfg))
    rollout_dir = Path(cfg.output_root) / cfg.experiment_name / f"eval_{int(cfg.eval.num_rollouts)}rollouts_{int(cfg.eval.max_steps)}steps"
    video_dir = rollout_dir / f"videos_execute_{execute_steps}"
    videos_remaining = _resolve_video_count(cfg) if save_videos else 0
    video_offset = 0

    episode_successes: list[bool] = []
    reward_traces: list[list[float]] = []
    success_traces: list[list[bool]] = []
    episode_rewards: list[float] = []
    video_paths: list[str] = []

    for start in range(0, len(eval_episodes), rollout_batch_size):
        batch_slice = slice(start, min(start + rollout_batch_size, len(eval_episodes)))
        batch = _run_world_batch(
            system,
            cfg,
            dataset=dataset,
            episodes_idx=eval_episodes[batch_slice],
            start_steps=eval_starts[batch_slice],
            execute_steps=execute_steps,
            conditioning_mode=conditioning_mode,
            save_video_count=videos_remaining,
            video_dir=video_dir,
            video_offset=video_offset,
        )
        episode_successes.extend(bool(value) for value in batch["episode_successes"])
        reward_traces.extend(batch["reward_traces"])
        success_traces.extend(batch["success_traces"])
        episode_rewards.extend(float(value) for value in batch["episode_rewards"])
        video_paths.extend(batch["video_paths"])
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
    }


def main() -> None:
    args, cfg = load_cfg_for_eval()
    system = VW2DirectActSystem(cfg, "joint")
    system.load_weights_from_checkpoint(args.checkpoint, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system.to(device)
    system.eval()

    conditioning_mode = _resolve_conditioning_mode(cfg)
    offline_metrics = _offline_metrics(system, cfg, conditioning_mode=conditioning_mode)
    execute_sweep = _resolve_execute_sweep(cfg)
    output_dir = Path(cfg.output_root) / cfg.experiment_name / f"eval_{int(cfg.eval.num_rollouts)}rollouts_{int(cfg.eval.max_steps)}steps"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for execute_steps in execute_sweep:
        world_metrics = _world_metrics(
            system,
            cfg,
            conditioning_mode=conditioning_mode,
            execute_steps=execute_steps,
            save_videos=bool(cfg.eval.save_video) and execute_steps == execute_sweep[0],
        )
        payload = {
            "experiment_name": str(cfg.experiment_name),
            "conditioning_mode": conditioning_mode,
            "execute_actions_per_plan": int(execute_steps),
            "offline": offline_metrics,
            "world": world_metrics,
        }
        all_results.append(payload)
        with (output_dir / f"execute_{execute_steps}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
