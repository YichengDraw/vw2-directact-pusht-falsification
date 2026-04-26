# Round 2 Oracle-Fix Artifacts

Final public artifacts for `pusht_subgoal_distill_round2_oraclefix`.

## Contents

- `subgoal_distill_round2_oraclefix_report.pdf` and `.tex`
- `eval_subgoal_50rollouts_100steps/summary.json`
- BC and TeacherOracle per-episode CSVs
- rollout videos for direct-act oracle sanity and TeacherOracle execute-1
- `teacher_oracle/` config, hparams, and CSV metrics
- `directact_oracle_eval_50rollouts_100steps/` sanity-check JSONs

## Scope

World rollouts are deterministic starts from `pusht_expert_train.h5`. BC and direct-act oracle sanity use episodes 0-49 at `start_step=0`; TeacherOracle uses episodes 101-150 at `start_step=3`. Offline splits are window-level splits.

The saved training config preserves local provenance paths. Use the root README for portable commands. Final world evaluation used `eval.rollout_batch_size=10`, which changes memory use only.

## Verdict

Gate A failed after the oracle-conditioning fix: TeacherOracle reached 0.0% success on execute-1 and execute-2. StudentFrozen and StudentJoint were stopped.

Checkpoints, raw datasets, local caches, and obsolete round1 artifacts are excluded.
