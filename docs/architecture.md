# VW2-DirectAct Push-T Architecture

## Model Pipeline

![VW2-DirectAct continuous-subgoal model pipeline](figures/model_pipeline.svg)

The subgoal branch has two conditioning paths:

- `TeacherOracle`: encodes a future rollout window with `FutureBottleneck` and feeds the continuous subgoal to `ActionDecoder`.
- `Student`: predicts the continuous subgoal from encoded history with `SubgoalPredictor`, then feeds it to the same `ActionDecoder`.

The oracle-fix rerun advances the oracle future window at every rollout step, which removes the stale step-0 oracle-subgoal failure mode.

## Experiment Flow

![Push-T oracle-fix experiment flow](figures/experiment_flow.svg)

Gate A is the hard stop: if TeacherOracle fails, StudentFrozen and StudentJoint are skipped. See the final report for metrics and artifacts.
