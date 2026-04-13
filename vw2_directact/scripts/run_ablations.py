from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGE_MODULES = {
    "tokenizer": "vw2_directact.train.train_tokenizer",
    "planner": "vw2_directact.train.train_planner",
    "action": "vw2_directact.train.train_action_decoder",
    "joint": "vw2_directact.train.train_joint",
}


def _last_checkpoint(root: Path, experiment_name: str, stage: str) -> Path:
    path = root / experiment_name / stage / "last.ckpt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def _run_module(module: str, *, python_exec: str, config_name: str, config_path: str | None, overrides: list[str]) -> None:
    command = [python_exec, "-m", module, "--config-name", config_name]
    if config_path:
        command.extend(["--config-path", config_path])
    command.extend(overrides)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BC, no-VQ, no-consistency, and full VW2-DirectAct experiments.")
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_root = Path("./vw2_directact_outputs")
    variants = [
        {
            "name": "bc",
            "stages": ["action"],
            "overrides": ["ablation.mode=bc", "model.use_vq=false", "loss.consistency_weight=0.0"],
        },
        {
            "name": "no_vq",
            "stages": ["tokenizer", "planner", "action", "joint"],
            "overrides": ["ablation.mode=no_vq", "model.use_vq=false"],
        },
        {
            "name": "no_consistency",
            "stages": ["tokenizer", "planner", "action", "joint"],
            "overrides": ["ablation.mode=no_consistency", "loss.consistency_weight=0.0"],
        },
        {
            "name": "full",
            "stages": ["tokenizer", "planner", "action", "joint"],
            "overrides": ["ablation.mode=full", "model.use_vq=true", "loss.consistency_weight=0.5"],
        },
    ]

    for variant in variants:
        experiment_name = f"{args.config_name}_{variant['name']}"
        init_from: Path | None = None
        for stage in variant["stages"]:
            overrides = [f"experiment_name={experiment_name}"] + list(variant["overrides"])
            if init_from is not None:
                overrides.append(f"train.init_from={init_from.as_posix()}")
            _run_module(
                STAGE_MODULES[stage],
                python_exec=args.python,
                config_name=args.config_name,
                config_path=args.config_path,
                overrides=overrides,
            )
            init_from = _last_checkpoint(output_root, experiment_name, stage)

        if args.skip_eval:
            continue
        _run_module(
            "vw2_directact.train.eval_policy",
            python_exec=args.python,
            config_name=args.config_name,
            config_path=args.config_path,
            overrides=[
                "--checkpoint",
                init_from.as_posix(),
                f"experiment_name={experiment_name}",
                *variant["overrides"],
            ],
        )


if __name__ == "__main__":
    main()
