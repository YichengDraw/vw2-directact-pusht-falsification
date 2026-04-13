from .subgoal_common import run_subgoal_stage


if __name__ == "__main__":
    run_subgoal_stage("student_predictor", description="Train the frozen-actor student predictor.")
