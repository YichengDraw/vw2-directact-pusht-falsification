# Push-T Subgoal Distillation Round 2 Oracle-Fix Artifacts

This folder contains the final public artifacts from the `pusht_subgoal_distill_round2_oraclefix` rerun.

## Included

- `subgoal_distill_round2_oraclefix_report.tex` and `subgoal_distill_round2_oraclefix_report.pdf`
- `eval_subgoal_50rollouts_100steps/summary.json`
- BC and TeacherOracle per-episode CSVs
- 10 BC rollout videos
- 10 TeacherOracle rollout videos
- TeacherOracle config, environment snapshot, frozen requirements, and CSV metrics
- direct-act oracle sanity-check evaluation JSONs after the evaluator fix

## Excluded

- training checkpoints
- raw datasets
- local caches

This is the final rerun after the stepwise oracle-conditioning fix. Gate A still failed, so the Push-T future-conditioned branch was stopped.
