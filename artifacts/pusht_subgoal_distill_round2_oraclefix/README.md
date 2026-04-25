# Push-T Subgoal Distillation Round 2 Oracle-Fix Artifacts

This folder contains the final public artifacts from the `pusht_subgoal_distill_round2_oraclefix` rerun.

## Included

- `subgoal_distill_round2_oraclefix_report.tex` and `subgoal_distill_round2_oraclefix_report.pdf`
- `eval_subgoal_50rollouts_100steps/summary.json`
- BC and TeacherOracle per-episode CSVs
- 10 direct-act oracle sanity-check rollout videos under `eval_50rollouts_100steps/videos_execute_1/`
- 10 TeacherOracle execute-1 rollout videos under `eval_subgoal_50rollouts_100steps/TeacherOracle/videos_execute_1/`
- TeacherOracle config, environment snapshot, frozen requirements, and CSV metrics
- direct-act oracle sanity-check evaluation JSONs after the evaluator fix

`teacher_oracle/config.yaml` is the saved training-stage config. The final world-evaluation commands used the command-line override `eval.rollout_batch_size=10`; batching changes memory use only and does not change rollout semantics.

The saved training config and Lightning hparams preserve original local provenance paths such as `train.init_from` and the dataset cache. Use the portable commands in the repository README for reproduction.

## Evaluation Scope

The world rollouts are deterministic starts from `pusht_expert_train.h5`, not an episode-held-out test split. BC and the direct-act oracle sanity check use direct-act evaluator starts: episodes 0-49 at `start_step=0`. TeacherOracle uses subgoal evaluator starts: episodes 101-150 at `start_step=3`. Offline train/validation samples are window-level splits, so they should not be interpreted as episode-level generalization evidence.

## Excluded

- training checkpoints
- raw datasets
- local caches

This is the only published experiment artifact set in the final package. The older first-pass `round1` artifacts were removed because they predate the stepwise oracle-conditioning fix.

Gate A still failed after the fix, so the Push-T future-conditioned branch was stopped before StudentFrozen or StudentJoint.
