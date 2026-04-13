from .subgoal_common import run_subgoal_stage


if __name__ == "__main__":
    run_subgoal_stage("teacher_oracle", description="Train the oracle teacher with future subgoals.")
