from .helpers import setup_pipeline, log_separator
from .eda import run_eda_conditions, run_eda_medications, run_eda_heart_failure
from .preprocess_data import build_t2dm_cohort, build_metformin_cohort, build_no_prior_hf_cohort


def main():
    setup_pipeline()

    log_separator("Starting pipeline")

    log_separator("EDA: T2DM identification reasoning")
    run_eda_conditions()

    log_separator("Preprocessing: building T2DM cohort")
    build_t2dm_cohort()

    log_separator("EDA: Antidiabetic medications reasoning")
    run_eda_medications()

    log_separator("Preprocessing: building metformin monotherapy cohort")
    build_metformin_cohort()

    log_separator("EDA: Heart failure identification reasoning")
    run_eda_heart_failure()

    log_separator("Preprocessing: excluding patients with prior heart failure")
    build_no_prior_hf_cohort()

    log_separator("Pipeline finished")


if __name__ == "__main__":
    main()
