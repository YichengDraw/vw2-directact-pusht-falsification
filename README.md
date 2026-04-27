# VW2-DirectAct Push-T Falsification

Final public evidence package for the Push-T continuous-subgoal branch after the rollout-side oracle-conditioning fix.

**Innovation tested:** a future-conditioned TeacherOracle compresses rollout information into a continuous subgoal; a history-only Student is meant to predict that subgoal and reuse the same action decoder.

**Verdict:** TeacherOracle failed Gate A after the fixed rerun: **0.0% world success on execute-1** and **0.0% world success on execute-2**. StudentFrozen and StudentJoint were stopped.

![VW2-DirectAct continuous-subgoal distillation architecture](docs/figures/pusht_falsification_overview.png)

Final materials:

- Report: `artifacts/pusht_subgoal_distill_round2_oraclefix/subgoal_distill_round2_oraclefix_report.pdf`
- Report source: `artifacts/pusht_subgoal_distill_round2_oraclefix/subgoal_distill_round2_oraclefix_report.tex`
- Artifacts: `artifacts/pusht_subgoal_distill_round2_oraclefix/`
- Architecture: `docs/architecture.md`

## Layout

```text
vw2_directact/   training, evaluation, model, data, and test code
docs/            architecture notes and figures
artifacts/       final oracle-fix evidence package
```

Checkpoints, raw datasets, and local caches are excluded.

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

## Final Rerun

The final world evaluation used `eval.rollout_batch_size=10` for RTX 4070 8 GB stability. This changes batching only.

```powershell
python -m vw2_directact.train.train_teacher_oracle --config-name pusht `
  experiment_name=pusht_subgoal_distill_round2_oraclefix `
  train.init_from=/path/to/pusht_falsification_oracle/action/last.ckpt

python -m vw2_directact.train.eval_subgoal_policy --config-name pusht `
  --bc-checkpoint /path/to/pusht_falsification_bc/action/last.ckpt `
  --teacher-checkpoint ./vw2_directact_outputs/pusht_subgoal_distill_round2_oraclefix/teacher_oracle/last.ckpt `
  experiment_name=pusht_subgoal_distill_round2_oraclefix `
  eval.rollout_batch_size=10 `
  eval.num_rollouts=50 `
  eval.max_steps=100

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

## Validation

```powershell
python -B -m compileall -q vw2_directact
python -B -m unittest discover -s vw2_directact\tests -v
git diff --check
```

World evaluation raises a hard error when `eval.run_world=true` but `stable_worldmodel`, the Push-T HDF5 dataset, or valid rollout starts are unavailable.

## License

MIT
