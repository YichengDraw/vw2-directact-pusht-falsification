# VW2-DirectAct Push-T Architecture

This repository is the final public evidence package for the Push-T continuous-subgoal branch after the oracle-conditioning fix.

## Model Pipeline

![VW2-DirectAct continuous-subgoal model pipeline](figures/model_pipeline.svg)

The subgoal branch has two conditioning paths:

- `TeacherOracle`: encodes the future rollout window with `FutureBottleneck` and feeds the resulting continuous subgoal to `ActionDecoder`.
- `Student`: predicts a continuous subgoal from encoded history with `SubgoalPredictor`, then feeds it to the same `ActionDecoder`.

The oracle-fix rerun advances the oracle future window at every world rollout step. This prevents the old failure mode where the evaluator reused the step-0 oracle subgoal for the whole episode.

## Experiment Flow

![Push-T oracle-fix experiment flow](figures/experiment_flow.svg)

Gate A requires TeacherOracle world success of at least 90% on execute-1 and execute-2. The final rerun reached 0.0% on both settings, so StudentFrozen and StudentJoint were not run.

The direct-act oracle sanity check reached 100.0% on execute-1 and 98.0% on execute-2 after the same evaluator fix, which confirms that the evaluator can still recognize a strong oracle policy.

The world rollouts are deterministic starts from `pusht_expert_train.h5`, not an episode-held-out test split. BC and the direct-act oracle sanity check use direct-act evaluator starts: episodes 0-49 at `start_step=0`. TeacherOracle uses subgoal evaluator starts: episodes 101-150 at `start_step=3`.

## Final Public Artifacts

The final evidence lives under:

```text
artifacts/pusht_subgoal_distill_round2_oraclefix/
```

That folder contains the report, summary JSON, per-episode CSVs, rollout videos, TeacherOracle training logs, and direct-act oracle sanity-check JSONs. Older first-pass artifacts were removed from the public package to keep the repository focused on the validated oracle-fix result.
