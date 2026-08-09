from .helpers import setup_pipeline, log_separator
from .eda import run_eda


def main():
    setup_pipeline()

    log_separator("Starting pipeline")

    log_separator("EDA: T2DM identification reasoning")
    run_eda()

    log_separator("Pipeline finished")


if __name__ == "__main__":
    main()