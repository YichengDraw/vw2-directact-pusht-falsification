from .common import run_stage


if __name__ == "__main__":
    run_stage("joint", description="Jointly train planner and action decoder with scheduled sampling.")
