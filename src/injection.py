import logging
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .config import (
    COMPLETE_CASE_COHORT_PATH,
    TREATMENT_ASSIGNMENT_OUTPUT_PATH,
    HF_OUTCOME_OUTPUT_PATH,
    RANDOM_SEED,
    ASSIGNMENT_COEFFICIENTS,
    COVARIATE_COLUMNS,
    TRUE_SGLT2I_HAZARD_RATIO,
    TRUE_SGLT2I_LOG_HR,
    OUTCOME_COEFFICIENTS,
    FOLLOWUP_DAYS,
    TARGET_OVERALL_EVENT_RATE,
)
from .helpers import log_separator


def bisect_calibrate(evaluate_fn, target_rate: float, low: float, high: float, n_iter: int = 50) -> float:
    """ Binary-searches for the parameter value whose evaluate_fn output has a mean equal to target_rate """
    for _ in range(n_iter):
        mid = (low + high) / 2
        if evaluate_fn(mid).mean() < target_rate:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def standardize_covariates(cohort: pd.DataFrame) -> pd.DataFrame:
    """ Z-score each covariate using the cohort's own observed mean/std """
    standardized = cohort.copy()

    for col in COVARIATE_COLUMNS:
        mean = cohort[col].mean()
        std = cohort[col].std()
        standardized[col] = (cohort[col] - mean) / std
        logging.info(f"  {col}: mean={mean:.2f}, std={std:.2f}")
    
    return standardized


def compute_sglt2i_probability(standardized: pd.DataFrame, intercept: float) -> pd.Series:
    """ Logistic combination of standardized covariates -> P(SGLT2i) """
    linear_predictor = pd.Series(intercept, index=standardized.index)
    for col, coef in ASSIGNMENT_COEFFICIENTS.items():
        linear_predictor = linear_predictor + coef * standardized[col]

    probability = 1 / (1 + np.exp(-linear_predictor))

    return probability


def calibrate_intercept(standardized: pd.DataFrame, target_rate: float = 0.45) -> float:
    """ Finds the intercept that makes the COHORT-AVERAGE P(SGLT2i) equal to target_rate by simple bisection search """
    return bisect_calibrate(
        lambda intercept: compute_sglt2i_probability(standardized, intercept),
        target_rate, -10.0, 10.0,
    )


def assign_treatment(cohort: pd.DataFrame) -> pd.DataFrame:
    """ Draws each patient's injected treatment (SGLT2i vs. DPP-4i) from the calibrated assignment model """
    logging.info("Standardizing covariates for treatment-assignment model ...")
    standardized = standardize_covariates(cohort)

    logging.info("Calibrating intercept to target ~45% overall SGLT2i assignment rate ...")
    intercept = calibrate_intercept(standardized, target_rate=0.45)
    logging.info(f"  Calibrated intercept: {intercept:.4f}")

    probability = compute_sglt2i_probability(standardized, intercept)

    rng = np.random.default_rng(RANDOM_SEED)
    random_draw = rng.uniform(size=len(cohort))
    treatment = np.where(random_draw < probability, "SGLT2i", "DPP4i")

    result = cohort[["patient_id", "metformin_start_date"]].copy()
    result["sglt2i_probability"] = probability.values
    result["treatment"] = treatment

    n_sglt2i = (treatment == "SGLT2i").sum()
    n_dpp4i = (treatment == "DPP4i").sum()
    logging.info(f"  Assigned: {n_sglt2i:,} SGLT2i ({n_sglt2i/len(cohort):.1%}), {n_dpp4i:,} DPP-4i ({n_dpp4i/len(cohort):.1%})")

    return result


def check_covariate_balance_by_treatment_arm(cohort: pd.DataFrame, assignment: pd.DataFrame) -> None:
    """ Computes each covariate's mean by treatment arm, plus the standardized mean difference between arms """
    merged = cohort.merge(assignment[["patient_id", "treatment"]], on="patient_id")
 
    sglt2i = merged.loc[merged["treatment"] == "SGLT2i"]
    dpp4i = merged.loc[merged["treatment"] == "DPP4i"]

    smds = {}
    logging.info(f"Covariate balance by treatment arm (SGLT2i n={len(sglt2i):,}, DPP-4i n={len(dpp4i):,}):")
    for col in COVARIATE_COLUMNS:
        mean_sglt2i = sglt2i[col].mean()
        mean_dpp4i = dpp4i[col].mean()
        pooled_std = np.sqrt((sglt2i[col].var() + dpp4i[col].var()) / 2)
        smd = (mean_sglt2i - mean_dpp4i) / pooled_std if pooled_std > 0 else 0.0
        smds[col] = smd
 
        flag = "  <- imbalanced (|SMD| > 0.1), confounding present as intended" if abs(smd) > 0.1 else ""
        logging.info(f"  {col}: SGLT2i mean={mean_sglt2i:.2f}, DPP-4i mean={mean_dpp4i:.2f}, SMD={smd:+.3f}{flag}")

    return smds


def build_treatment_assignment():
    """ Loads the complete-case cohort, injects treatment assignment, and saves it with a balance check """
    logging.info(f"Loading complete-case cohort from {COMPLETE_CASE_COHORT_PATH} ...")
    cohort = pd.read_csv(COMPLETE_CASE_COHORT_PATH, parse_dates=["metformin_start_date"])
    logging.info(f"  {len(cohort):,} patients loaded")
    log_separator()

    logging.info("Assigning injected treatment (SGLT2i vs. DPP-4i) ...")
    assignment = assign_treatment(cohort)

    assignment.to_csv(TREATMENT_ASSIGNMENT_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {TREATMENT_ASSIGNMENT_OUTPUT_PATH}")
    log_separator()

    check_covariate_balance_by_treatment_arm(cohort, assignment)
 
 
def compute_hazard(standardized: pd.DataFrame, treatment: pd.Series, log_lambda_0: float) -> pd.Series:
    """ Computes the hazard for each patient given standardized covariates, treatment assignment, and log-baseline-hazard """
    linear_predictor = pd.Series(log_lambda_0, index=standardized.index)
    linear_predictor = linear_predictor + TRUE_SGLT2I_LOG_HR * (treatment == "SGLT2i").astype(float)
    for col, coef in OUTCOME_COEFFICIENTS.items():
        linear_predictor = linear_predictor + coef * standardized[col]
 
    return np.exp(linear_predictor)
 
 
def calibrate_baseline_hazard(standardized: pd.DataFrame, treatment: pd.Series, target_rate: float) -> float:
    """ Finds the log-baseline-hazard that makes the COHORT-AVERAGE 2-year cumulative incidence equal to target_rate by bisection """
    def cumulative_incidence(log_lambda_0):
        hazard = compute_hazard(standardized, treatment, log_lambda_0)
        return 1 - np.exp(-hazard * FOLLOWUP_DAYS)

    return bisect_calibrate(cumulative_incidence, target_rate, -20.0, 5.0)
 
 
def simulate_hf_outcome(cohort: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """ Simulates each patient's injected HF hospitalization event time and censoring status """
    merged = cohort.merge(assignment[["patient_id", "treatment"]], on="patient_id")
 
    logging.info("Standardizing covariates for outcome model ...")
    standardized = standardize_covariates(merged)
 
    logging.info(f"Calibrating baseline hazard to target {TARGET_OVERALL_EVENT_RATE:.0%} cumulative incidence over {FOLLOWUP_DAYS} days ...")
    log_lambda_0 = calibrate_baseline_hazard(standardized, merged["treatment"], TARGET_OVERALL_EVENT_RATE)
    logging.info(f"  Calibrated log-baseline-hazard: {log_lambda_0:.4f}")
 
    hazard = compute_hazard(standardized, merged["treatment"], log_lambda_0)

    rng = np.random.default_rng(RANDOM_SEED + 1)
    uniform_draw = rng.uniform(size=len(merged))

    # Inverse-CDF sampling for exponential event times: T = -ln(U) / hazard
    simulated_event_days = -np.log(uniform_draw) / hazard
 
    observed_days = np.minimum(simulated_event_days, FOLLOWUP_DAYS)
    event_occurred = simulated_event_days <= FOLLOWUP_DAYS
 
    result = merged[["patient_id", "metformin_start_date", "treatment"]].copy()
    result["hf_event_days"] = observed_days
    result["hf_event_occurred"] = event_occurred
 
    n_events = event_occurred.sum()
    logging.info(f"  {n_events:,}/{len(result):,} patients ({n_events/len(result):.1%}) experienced the injected HF event within {FOLLOWUP_DAYS} days")
 
    for arm in ["SGLT2i", "DPP4i"]:
        arm_mask = result["treatment"] == arm
        arm_rate = result.loc[arm_mask, "hf_event_occurred"].mean()
        logging.info(f"  {arm}: {arm_rate:.1%} event rate (n={arm_mask.sum():,})")
 
    return result


def check_naive_treatment_effect(outcome: pd.DataFrame) -> dict:
    """ Fits a simple unadjusted Cox proportional hazards model with treatment as the only covariate to quantify the naive confounding bias """
    cox_data = outcome[["hf_event_days", "hf_event_occurred", "treatment"]].copy()
    cox_data["sglt2i"] = (cox_data["treatment"] == "SGLT2i").astype(int)
    cox_data = cox_data.drop(columns="treatment")
 
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="hf_event_days", event_col="hf_event_occurred")
 
    naive_hr = np.exp(cph.params_["sglt2i"])
    ci_low, ci_high = np.exp(cph.confidence_intervals_.loc["sglt2i"])
 
    logging.info("Unadjusted (naive) Cox model - treatment as sole covariate:")
    logging.info(f"  Naive HR (SGLT2i vs. DPP-4i): {naive_hr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
    logging.info(f"  TRUE injected HR: {TRUE_SGLT2I_HAZARD_RATIO:.3f}")
    logging.info(f"  Naive estimate is {'biased away from' if naive_hr < TRUE_SGLT2I_HAZARD_RATIO else 'biased toward null relative to'} the true effect by confounding")
 
    return {"model": "Naive (unadjusted)", "hr": naive_hr, "ci_low": ci_low, "ci_high": ci_high}
 
 
def build_hf_outcome():
    """ Loads the cohort and treatment assignment, injects the HF outcome, and saves it with a naive-effect check """
    logging.info(f"Loading complete-case cohort from {COMPLETE_CASE_COHORT_PATH} ...")
    cohort = pd.read_csv(COMPLETE_CASE_COHORT_PATH, parse_dates=["metformin_start_date"])
    logging.info(f"  {len(cohort):,} patients loaded")
 
    logging.info(f"Loading treatment assignment from {TREATMENT_ASSIGNMENT_OUTPUT_PATH} ...")
    assignment = pd.read_csv(TREATMENT_ASSIGNMENT_OUTPUT_PATH)
    logging.info(f"  {len(assignment):,} patients loaded")
    log_separator()
 
    logging.info("Simulating injected HF hospitalization outcome ...")
    outcome = simulate_hf_outcome(cohort, assignment)
    outcome.to_csv(HF_OUTCOME_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {HF_OUTCOME_OUTPUT_PATH}")
    log_separator()
    
    logging.info("Fitting unadjusted Cox model with treatment as sole covariate to quantify naive confounding bias ...")
    check_naive_treatment_effect(outcome)
