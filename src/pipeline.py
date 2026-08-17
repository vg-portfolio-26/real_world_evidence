from .helpers import setup_pipeline, log_separator
from .eda import run_eda_conditions, run_eda_medications, run_eda_heart_failure, run_eda_observations, describe_complete_case_cohort_for_injection_calibration
from .preprocess_data import build_t2dm_cohort, build_metformin_cohort, build_no_prior_hf_cohort, build_baseline_covariates, build_complete_case_cohort


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

    log_separator("EDA: Baseline covariate observations reasoning")
    run_eda_observations()

    log_separator("Preprocessing: extracting baseline covariates")
    build_baseline_covariates()

    log_separator("EDA: Complete-case cohort descriptive stats")
    describe_complete_case_cohort_for_injection_calibration()

    log_separator("Preprocessing: building complete-case cohort")
    build_complete_case_cohort()

    log_separator("Pipeline finished")


if __name__ == "__main__":
    main()
