from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


STAGE_MODULES = {
    "tokenizer": "vw2_directact.train.train_tokenizer",
    "planner": "vw2_directact.train.train_planner",
    "action": "vw2_directact.train.train_action_decoder",
    "joint": "vw2_directact.train.train_joint",
}

EXECUTE_SWEEP = (1, 2, 4)


def _run_module(module: str, *, python_exec: str, config_name: str, config_path: str | None, overrides: list[str]) -> None:
    command = [python_exec, "-m", module, "--config-name", config_name]
    if config_path:
        command.extend(["--config-path", config_path])
    command.extend(overrides)
    subprocess.run(command, check=True)


def _last_checkpoint(root: Path, experiment_name: str, stage: str) -> Path:
    stage_dir = root / experiment_name / stage
    last_path = stage_dir / "last.ckpt"
    if last_path.exists():
        return last_path
    candidates = sorted(stage_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing checkpoint under: {stage_dir}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_world_success(records: list[dict]) -> tuple[float, int]:
    best = max(records, key=lambda record: float(record["world"]["success_rate"]))
    return float(best["world"]["success_rate"]), int(best["execute_actions_per_plan"])


def _gate_b_passes(diagnostics: dict) -> tuple[bool, float]:
    chance = float(diagnostics["chance_token_acc"])
    threshold = max(chance * 5.0, chance + 0.02)
    shuffled = float(diagnostics["token_acc_shuffled_future_targets"])
    return shuffled <= threshold, threshold


def _stage_checkpoint_if_complete(root: Path, experiment_name: str, stage: str) -> Path | None:
    stage_dir = root / experiment_name / stage
    if not stage_dir.exists():
        return None
    try:
        return _last_checkpoint(root, experiment_name, stage)
    except FileNotFoundError:
        return None


def _eval_records_if_complete(output_root: Path, experiment_name: str) -> list[dict] | None:
    eval_dir = output_root / experiment_name / "eval_50rollouts_100steps"
    required = [eval_dir / f"execute_{execute_steps}.json" for execute_steps in EXECUTE_SWEEP]
    if not all(path.exists() for path in required):
        return None
    return [_load_json(path) for path in required]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy VW2-DirectAct Push-T sweep. The public final package "
            "uses the oracle-fix subgoal rerun under artifacts/pusht_subgoal_distill_round2_oraclefix."
        )
    )
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true", help="Rerun stages even when outputs already exist.")
    args = parser.parse_args()

    output_root = Path("./vw2_directact_outputs")
    round_root = output_root / "pusht_falsification_round"
    round_root.mkdir(parents=True, exist_ok=True)

    common_overrides = [
        "train.max_epochs=3",
        "train.limit_train_batches=200",
        "train.limit_val_batches=20",
        "data.max_train_samples=12000",
        "data.max_val_samples=1024",
        "eval.num_rollouts=50",
        "eval.max_steps=100",
        "eval.rollout_batch_size=10",
    ]

    experiments = [
        {
            "label": "BC",
            "experiment_name": "pusht_falsification_bc",
            "stages": ["action"],
            "overrides": [
                "ablation.mode=bc",
                "conditioning.mode=bc",
                "model.use_vq=false",
                "loss.consistency_weight=0.0",
                "eval.save_video=false",
            ],
            "diagnose_planner": False,
        },
        {
            "label": "OracleFuture",
            "experiment_name": "pusht_falsification_oracle",
            "stages": ["tokenizer", "action"],
            "overrides": [
                "ablation.mode=full",
                "conditioning.mode=oracle",
                "model.use_vq=true",
                "loss.consistency_weight=0.5",
                "eval.save_video=false",
            ],
            "diagnose_planner": False,
        },
        {
            "label": "PredFuture(full)",
            "experiment_name": "pusht_falsification_predfuture",
            "stages": ["tokenizer", "planner", "action", "joint"],
            "overrides": [
                "ablation.mode=full",
                "conditioning.mode=predfuture",
                "model.use_vq=true",
                "loss.consistency_weight=0.5",
                "eval.save_video=true",
            ],
            "diagnose_planner": True,
        },
    ]

    summary_rows = []
    gate_inputs: dict[str, dict] = {}

    for experiment in experiments:
        init_from: Path | None = None
        for stage in experiment["stages"]:
            existing_checkpoint = None if args.force else _stage_checkpoint_if_complete(output_root, experiment["experiment_name"], stage)
            if existing_checkpoint is not None:
                init_from = existing_checkpoint
                continue
            overrides = [f"experiment_name={experiment['experiment_name']}", *common_overrides, *experiment["overrides"]]
            if init_from is not None:
                overrides.append(f"train.init_from={init_from.as_posix()}")
            _run_module(
                STAGE_MODULES[stage],
                python_exec=args.python,
                config_name=args.config_name,
                config_path=args.config_path,
                overrides=overrides,
            )
            init_from = _last_checkpoint(output_root, experiment["experiment_name"], stage)

        existing_records = None if args.force else _eval_records_if_complete(output_root, experiment["experiment_name"])
        if existing_records is None:
            eval_overrides = [
                "--checkpoint",
                init_from.as_posix(),
                f"experiment_name={experiment['experiment_name']}",
                *common_overrides,
                *experiment["overrides"],
            ]
            _run_module(
                "vw2_directact.train.eval_policy",
                python_exec=args.python,
                config_name=args.config_name,
                config_path=args.config_path,
                overrides=eval_overrides,
            )

        planner_diag_path = output_root / experiment["experiment_name"] / "planner_diagnostics.json"
        if experiment["diagnose_planner"] and (args.force or not planner_diag_path.exists()):
            _run_module(
                "vw2_directact.train.diagnose_planner",
                python_exec=args.python,
                config_name=args.config_name,
                config_path=args.config_path,
                overrides=[
                    "--checkpoint",
                    init_from.as_posix(),
                    f"experiment_name={experiment['experiment_name']}",
                    *common_overrides,
                    *experiment["overrides"],
                ],
            )

        eval_dir = output_root / experiment["experiment_name"] / "eval_50rollouts_100steps"
        records = existing_records if existing_records is not None else [_load_json(eval_dir / f"execute_{execute_steps}.json") for execute_steps in EXECUTE_SWEEP]
        gate_inputs[experiment["label"]] = {"records": records, "checkpoint": str(init_from)}
        for record in records:
            summary_rows.append(
                {
                    "experiment": experiment["label"],
                    "experiment_name": experiment["experiment_name"],
                    "execute_actions_per_plan": int(record["execute_actions_per_plan"]),
                    "conditioning_mode": record["conditioning_mode"],
                    "offline_action_mse": float(record["offline"]["action_mse"]),
                    "offline_latency_ms": float(record["offline"]["latency_ms"]),
                    "world_success_rate": float(record["world"]["success_rate"]),
                    "world_mean_episode_reward": float(record["world"]["mean_episode_reward"]),
                    "json_path": str(eval_dir / f"execute_{record['execute_actions_per_plan']}.json"),
                }
            )

    summary_path = round_root / "ablation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    planner_diag_src = output_root / "pusht_falsification_predfuture" / "planner_diagnostics.json"
    planner_diag_dst = round_root / "planner_diagnostics.json"
    shutil.copy2(planner_diag_src, planner_diag_dst)
    diagnostics = _load_json(planner_diag_dst)

    pred_video_dir = output_root / "pusht_falsification_predfuture" / "eval_50rollouts_100steps" / "videos_execute_1"
    report_video_dir = round_root / "rollout_videos"
    report_video_dir.mkdir(parents=True, exist_ok=True)
    for mp4 in sorted(pred_video_dir.glob("*.mp4"))[:10]:
        shutil.copy2(mp4, report_video_dir / mp4.name)

    bc_success, bc_execute = _best_world_success(gate_inputs["BC"]["records"])
    oracle_success, oracle_execute = _best_world_success(gate_inputs["OracleFuture"]["records"])
    pred_success, pred_execute = _best_world_success(gate_inputs["PredFuture(full)"]["records"])
    gate_a = bc_success > 0.0
    gate_b, gate_b_threshold = _gate_b_passes(diagnostics)
    gate_c = oracle_success > bc_success

    passed_all = gate_a and gate_b and gate_c
    report_lines = [
        "# VW2-DirectAct Push-T Falsification Round",
        "",
        "## Experiments Run",
        "- BC",
        "- OracleFuture",
        "- PredFuture(full)",
        "",
        "No extra ablations were run beyond the requested falsification set.",
        "",
        "## Best World Success Across execute_actions_per_plan Sweep",
        f"- BC: {bc_success:.4f} success rate at execute_actions_per_plan={bc_execute}",
        f"- OracleFuture: {oracle_success:.4f} success rate at execute_actions_per_plan={oracle_execute}",
        f"- PredFuture(full): {pred_success:.4f} success rate at execute_actions_per_plan={pred_execute}",
        "",
        "## Planner Diagnostics",
        f"- chance_token_acc: {diagnostics['chance_token_acc']:.6f}",
        f"- token_acc_normal_conditioning: {diagnostics['token_acc_normal_conditioning']:.6f}",
        f"- token_acc_shuffled_future_targets: {diagnostics['token_acc_shuffled_future_targets']:.6f}",
        f"- token_acc_zeroed_current_observation: {diagnostics['token_acc_zeroed_current_observation']:.6f}",
        f"- token_entropy: {diagnostics['token_entropy']:.6f}",
        f"- codebook_perplexity: {diagnostics['codebook_perplexity']:.6f}",
        f"- top1_token_ratio: {diagnostics['top1_token_ratio']:.6f}",
        f"- unique_token_count: {diagnostics['unique_token_count']}",
        "",
        "## Gates",
        f"- Gate A: BC world success > 0 -> {'PASS' if gate_a else 'FAIL'} ({bc_success:.4f})",
        f"- Gate B: shuffled planner acc collapses toward chance -> {'PASS' if gate_b else 'FAIL'} (shuffled={diagnostics['token_acc_shuffled_future_targets']:.6f}, chance={diagnostics['chance_token_acc']:.6f}, threshold={gate_b_threshold:.6f})",
        f"- Gate C: OracleFuture beats BC -> {'PASS' if gate_c else 'FAIL'} (OracleFuture={oracle_success:.4f}, BC={bc_success:.4f})",
        "",
    ]
    if passed_all:
        report_lines.extend(
            [
                "## Verdict",
                "All gates passed in this falsification round.",
                "The branch is still viable on Push-T under the current test design.",
            ]
        )
    else:
        failure_reasons = []
        if not gate_a:
            failure_reasons.append("BC does not achieve any world success.")
        if not gate_b:
            failure_reasons.append("Shuffled planner accuracy does not collapse toward chance.")
        if not gate_c:
            failure_reasons.append("OracleFuture does not beat BC in world success.")
        report_lines.extend(
            [
                "## Verdict",
                "Hard stop engaged.",
                *[f"- {reason}" for reason in failure_reasons],
                "",
                "Recommendation: stop this branch on Push-T. Do not add new model capacity and do not run extra ablations on this task until the failed gate is explained.",
            ]
        )

    report_path = round_root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
