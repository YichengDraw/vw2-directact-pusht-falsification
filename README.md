# VW2-DirectAct Push-T Falsification

Final public evidence package for the Push-T continuous-subgoal branch after the rollout-side oracle-conditioning fix.

**Verdict:** TeacherOracle failed Gate A after the fixed rerun: **0.0% world success on execute-1** and **0.0% world success on execute-2**. StudentFrozen and StudentJoint were stopped.

## Overview

![VW2-DirectAct Push-T falsification overview](docs/figures/pusht_falsification_overview.png)

This overview is for orientation only. The committed source code, JSON/CSV artifacts, and final report are the evidence for metrics and conclusions.

Detailed architecture diagrams are in [docs/architecture.md](docs/architecture.md).

## Contents

```text
vw2_directact/   training, evaluation, model, data, and test code
docs/figures/   overview and architecture figures
artifacts/pusht_subgoal_distill_round2_oraclefix/
                 final report, summaries, CSVs, logs, sanity-check JSONs, videos
```

Checkpoints, raw datasets, and local caches are intentionally excluded.

## Setup

Use Python 3.11.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Push-T world rollouts require `stable_worldmodel` and `pusht_expert_train.h5`. The dataset resolves in this order:

1. `data.path=/absolute/path/to/pusht_expert_train.h5`
2. `$STABLEWM_HOME/pusht_expert_train.h5`
3. `~/.stable-wm/pusht_expert_train.h5`

## Final Rerun Commands

The final world evaluation used `eval.rollout_batch_size=10` for RTX 4070 8 GB stability. This changes batching only.

```powershell
python -m vw2_directact.train.train_teacher_oracle --config-name pusht `
  experiment_name=pusht_subgoal_distill_round2_oraclefix `
  train.init_from=/path/to/pusht_falsification_oracle/action/last.ckpt
```

```powershell
python -m vw2_directact.train.eval_subgoal_policy --config-name pusht `
  --bc-checkpoint /path/to/pusht_falsification_bc/action/last.ckpt `
  --teacher-checkpoint ./vw2_directact_outputs/pusht_subgoal_distill_round2_oraclefix/teacher_oracle/last.ckpt `
  experiment_name=pusht_subgoal_distill_round2_oraclefix `
  eval.rollout_batch_size=10 `
  eval.num_rollouts=50 `
  eval.max_steps=100
```

```powershell
python -m vw2_directact.train.eval_policy --config-name pusht `
  --checkpoint /path/to/pusht_falsification_oracle/action/last.ckpt `
  experiment_name=pusht_falsification_oracle `
  conditioning.mode=oracle `
  ablation.mode=full `
  model.use_vq=true `
  eval.rollout_batch_size=10 `
  eval.num_rollouts=50 `
  eval.max_steps=100
```

## Evaluation Scope

World rollouts are deterministic starts from `pusht_expert_train.h5`, selected from valid expert episodes. They are not an episode-held-out test split.

| Evaluator | Episodes | Start step |
| --- | ---: | ---: |
| BC / DirectAct oracle sanity | 0-49 | 0 |
| TeacherOracle subgoal | 101-150 | 3 |

Offline train/validation samples are window-level splits, so offline validation metrics are not episode-level generalization metrics.

## Results

| Model | Offline Action MSE | Execute-1 Success | Execute-2 Success | Execute-1 Reward | Execute-2 Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC | 0.022812 | 0.0% | 0.0% | -23644.98 | -24199.06 |
| TeacherOracle | 0.020557 | 0.0% | 0.0% | -21435.20 | -20406.16 |

TeacherOracle improved offline MSE and mean reward over BC, but produced zero successes in 100 world rollouts. Gate A required at least 90% success on execute-1 and execute-2, so the branch stopped.

Sanity check after the same evaluator fix:

| Model | Execute-1 Success | Execute-2 Success | Execute-4 Success |
| --- | ---: | ---: | ---: |
| DirectAct Oracle | 100.0% | 98.0% | 0.0% |

The sanity check confirms the evaluator still recognizes a strong oracle policy.

## Evidence

- Final report: `artifacts/pusht_subgoal_distill_round2_oraclefix/subgoal_distill_round2_oraclefix_report.pdf`
- Report source: `artifacts/pusht_subgoal_distill_round2_oraclefix/subgoal_distill_round2_oraclefix_report.tex`
- Summary JSON: `artifacts/pusht_subgoal_distill_round2_oraclefix/eval_subgoal_50rollouts_100steps/summary.json`
- Per-episode CSVs and videos: `artifacts/pusht_subgoal_distill_round2_oraclefix/eval_subgoal_50rollouts_100steps/`
- Direct-act oracle sanity JSONs: `artifacts/pusht_subgoal_distill_round2_oraclefix/directact_oracle_eval_50rollouts_100steps/`
- Teacher training logs: `artifacts/pusht_subgoal_distill_round2_oraclefix/teacher_oracle/`

## Validation

```powershell
python -B -m compileall -q vw2_directact
python -B -m unittest discover -s vw2_directact\tests -v
git diff --check
```

World evaluation raises a hard error when `eval.run_world=true` but `stable_worldmodel`, the Push-T HDF5 dataset, or valid rollout starts are unavailable.

## License

MIT
