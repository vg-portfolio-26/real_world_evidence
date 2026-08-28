import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from .helpers import ANALYSIS_DIR, log_separator
from .injection import (
    COMPLETE_CASE_COHORT_PATH,
    TREATMENT_ASSIGNMENT_OUTPUT_PATH,
    HF_OUTCOME_OUTPUT_PATH,
    COVARIATE_COLUMNS,
    OUTCOME_COEFFICIENTS,
    FOLLOWUP_DAYS,
    TARGET_OVERALL_EVENT_RATE,
    RANDOM_SEED,
    TRUE_SGLT2I_HAZARD_RATIO,
    standardize_covariates,
    check_covariate_balance_by_treatment_arm,
    check_naive_treatment_effect,
)

PROPENSITY_SCORES_OUTPUT_PATH = ANALYSIS_DIR / "propensity_scores.csv"
RESULTS_SUMMARY_OUTPUT_PATH = ANALYSIS_DIR / "results_summary.csv"
LOVE_PLOT_OUTPUT_PATH = ANALYSIS_DIR / "love_plot.png"
KM_CURVES_OUTPUT_PATH = ANALYSIS_DIR / "km_curves.png"
NEGATIVE_CONTROL_OUTPUT_PATH = ANALYSIS_DIR / "negative_control_results.csv"

NEGATIVE_CONTROL_TRUE_LOG_HR = 0.0


def load_analysis_dataset() -> pd.DataFrame:
    logging.info(f"Loading complete-case cohort from {COMPLETE_CASE_COHORT_PATH} ...")
    cohort = pd.read_csv(COMPLETE_CASE_COHORT_PATH, parse_dates=["metformin_start_date"])

    logging.info(f"Loading treatment assignment from {TREATMENT_ASSIGNMENT_OUTPUT_PATH} ...")
    assignment = pd.read_csv(TREATMENT_ASSIGNMENT_OUTPUT_PATH, usecols=["patient_id", "treatment"])

    logging.info(f"Loading HF outcome from {HF_OUTCOME_OUTPUT_PATH} ...")
    outcome = pd.read_csv(HF_OUTCOME_OUTPUT_PATH, usecols=["patient_id", "hf_event_days", "hf_event_occurred"])

    merged = cohort.merge(assignment, on="patient_id").merge(outcome, on="patient_id")
    logging.info(f"  {len(merged):,} patients in the merged analysis dataset")
    return merged


def estimate_propensity_scores(data: pd.DataFrame) -> pd.Series:
    """ Logistic regression predicting P(SGLT2i) from the same 6 baseline covariates used in the true assignment mechanism in src/injection.py """
    logging.info("Estimating propensity scores (logistic regression on 6 baseline covariates) ...")
    standardized = standardize_covariates(data)

    X = sm.add_constant(standardized[COVARIATE_COLUMNS])
    y = (data["treatment"] == "SGLT2i").astype(int)

    model = sm.Logit(y, X).fit(disp=0)
    logging.info("Propensity model coefficients (standardized covariates):")
    for col in COVARIATE_COLUMNS:
        logging.info(f"  {col}: coef={model.params[col]:+.3f}, p={model.pvalues[col]:.4f}")

    propensity_scores = model.predict(X)
    return propensity_scores


def compute_iptw_weights(propensity_scores: pd.Series, treatment: pd.Series) -> pd.Series:
    """ Compute stabilized IPTW weights for the average treatment effect """
    is_sglt2i = (treatment == "SGLT2i")
    marginal_p_sglt2i = is_sglt2i.mean()

    weights = pd.Series(index=treatment.index, dtype=float)
    weights[is_sglt2i] = marginal_p_sglt2i / propensity_scores[is_sglt2i]
    weights[~is_sglt2i] = (1 - marginal_p_sglt2i) / (1 - propensity_scores[~is_sglt2i])

    logging.info(f"  Stabilized IPTW weights computed: mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")

    return weights


def check_weighted_covariate_balance(data: pd.DataFrame, weights: pd.Series) -> dict:
    """ Check weighted SMD per covariate """
    is_sglt2i = (data["treatment"] == "SGLT2i")

    smds = {}
    logging.info("Weighted covariate balance after IPTW (compare to pre-weighting SMDs in injection step):")
    for col in COVARIATE_COLUMNS:
        w_sglt2i = weights[is_sglt2i]
        w_dpp4i = weights[~is_sglt2i]
        x_sglt2i = data.loc[is_sglt2i, col]
        x_dpp4i = data.loc[~is_sglt2i, col]

        weighted_mean_sglt2i = np.average(x_sglt2i, weights=w_sglt2i)
        weighted_mean_dpp4i = np.average(x_dpp4i, weights=w_dpp4i)

        weighted_var_sglt2i = np.average((x_sglt2i - weighted_mean_sglt2i) ** 2, weights=w_sglt2i)
        weighted_var_dpp4i = np.average((x_dpp4i - weighted_mean_dpp4i) ** 2, weights=w_dpp4i)
        pooled_std = np.sqrt((weighted_var_sglt2i + weighted_var_dpp4i) / 2)

        smd = (weighted_mean_sglt2i - weighted_mean_dpp4i) / pooled_std if pooled_std > 0 else 0.0
        smds[col] = smd
        flag = "  <- still imbalanced" if abs(smd) > 0.1 else "  <- balanced"
        logging.info(f"  {col}: SMD={smd:+.3f}{flag}")

    return smds


def fit_weighted_cox_model(data: pd.DataFrame, weights: pd.Series) -> dict:
    """ Re-fits the unadjusted-style Cox model from src/injection.py's check_naive_treatment_effect() with IPTW weights applied """
    cox_data = data[["hf_event_days", "hf_event_occurred", "treatment"]].copy()
    cox_data["sglt2i"] = (cox_data["treatment"] == "SGLT2i").astype(int)
    cox_data = cox_data.drop(columns="treatment")
    cox_data["weight"] = weights.values

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="hf_event_days", event_col="hf_event_occurred", weights_col="weight", robust=True)

    adjusted_hr = np.exp(cph.params_["sglt2i"])
    ci_low, ci_high = np.exp(cph.confidence_intervals_.loc["sglt2i"])

    logging.info("IPTW-weighted Cox model - treatment as sole covariate, weighted by stabilized IPTW:")
    logging.info(f"  Adjusted HR (SGLT2i vs. DPP-4i): {adjusted_hr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
    logging.info(f"  TRUE injected HR: {TRUE_SGLT2I_HAZARD_RATIO:.3f}")

    truth_in_ci = ci_low <= TRUE_SGLT2I_HAZARD_RATIO <= ci_high
    logging.info(
        f"  {'SUCCESS: true HR falls within the adjusted 95% CI' if truth_in_ci else 'NOTE: true HR still falls outside the adjusted 95% CI'} "
        f"- {'the propensity adjustment recovered the true effect' if truth_in_ci else 'residual bias remains, worth investigating (e.g. unmeasured confounding, model misspecification, or weight instability)'}"
    )

    return {"model": "IPTW-weighted", "hr": adjusted_hr, "ci_low": ci_low, "ci_high": ci_high}


def fit_doubly_robust_cox_model(data: pd.DataFrame, weights: pd.Series) -> dict:
    """ The same IPTW weights as fit_weighted_cox_model() plus the 6 baseline covariates included directly as regression adjustors """
    standardized = standardize_covariates(data)

    cox_data = data[["hf_event_days", "hf_event_occurred", "treatment"]].copy()
    cox_data["sglt2i"] = (cox_data["treatment"] == "SGLT2i").astype(int)
    cox_data = cox_data.drop(columns="treatment")
    cox_data["weight"] = weights.values
    for col in COVARIATE_COLUMNS:
        cox_data[col] = standardized[col].values

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="hf_event_days", event_col="hf_event_occurred", weights_col="weight", robust=True)

    adjusted_hr = np.exp(cph.params_["sglt2i"])
    ci_low, ci_high = np.exp(cph.confidence_intervals_.loc["sglt2i"])

    logging.info("Doubly-robust Cox model - IPTW weights PLUS covariate adjustment:")
    logging.info(f"  Adjusted HR (SGLT2i vs. DPP-4i): {adjusted_hr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
    logging.info(f"  TRUE injected HR: {TRUE_SGLT2I_HAZARD_RATIO:.3f}")

    truth_in_ci = ci_low <= TRUE_SGLT2I_HAZARD_RATIO <= ci_high
    logging.info(
        f"  {'SUCCESS: true HR falls within the doubly-robust 95% CI' if truth_in_ci else 'NOTE: true HR still falls outside the doubly-robust 95% CI'}"
    )

    return {"model": "Doubly-robust", "hr": adjusted_hr, "ci_low": ci_low, "ci_high": ci_high}


def build_results_table(naive_result: dict, iptw_result: dict, dr_result: dict) -> pd.DataFrame:
    """ Assembles the naive / IPTW / doubly-robust comparison table """
    rows = [naive_result, iptw_result, dr_result]
    table = pd.DataFrame(rows)
    table["true_hr"] = TRUE_SGLT2I_HAZARD_RATIO
    table["contains_truth"] = (table["ci_low"] <= TRUE_SGLT2I_HAZARD_RATIO) & (TRUE_SGLT2I_HAZARD_RATIO <= table["ci_high"])

    logging.info("Results summary table:")
    for _, row in table.iterrows():
        logging.info(
            f"  {row['model']}: HR={row['hr']:.3f} (95% CI: {row['ci_low']:.3f}-{row['ci_high']:.3f}), "
            f"contains truth ({row['true_hr']:.3f})? {row['contains_truth']}"
        )

    table.to_csv(RESULTS_SUMMARY_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {RESULTS_SUMMARY_OUTPUT_PATH}")
    return table


def plot_love_plot(unadjusted_smds: dict, weighted_smds: dict) -> None:
    """ SMD per covariate, before (unadjusted) vs. after (IPTW-weighted) """
    covariates = COVARIATE_COLUMNS
    y_positions = np.arange(len(covariates))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([unadjusted_smds[c] for c in covariates], y_positions, label="Before (unadjusted)", color="tab:red", zorder=3)
    ax.scatter([weighted_smds[c] for c in covariates], y_positions, label="After (IPTW)", color="tab:blue", zorder=3)

    for y, c in zip(y_positions, covariates):
        ax.plot([unadjusted_smds[c], weighted_smds[c]], [y, y], color="gray", linewidth=0.8, zorder=1)

    ax.axvline(0.1, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-0.1, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Standardized Mean Difference (SGLT2i vs. DPP-4i)")
    ax.set_title("Covariate Balance Before/After IPTW (Love Plot)")
    ax.legend()
    fig.tight_layout()

    fig.savefig(LOVE_PLOT_OUTPUT_PATH, dpi=150)
    plt.close(fig)
    logging.info(f"Saved: {LOVE_PLOT_OUTPUT_PATH}")


def plot_km_curves(data: pd.DataFrame) -> None:
    """ Unweighted Kaplan-Meier curves by treatment arm """
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, color in [("SGLT2i", "tab:blue"), ("DPP4i", "tab:orange")]:
        arm_data = data.loc[data["treatment"] == arm]
        kmf = KaplanMeierFitter()
        kmf.fit(arm_data["hf_event_days"], event_observed=arm_data["hf_event_occurred"], label=arm)
        kmf.plot_survival_function(ax=ax, color=color)

    ax.set_xlabel("Days since metformin start")
    ax.set_ylabel("HF-hospitalization-free survival probability")
    ax.set_title("Kaplan-Meier Curves by Treatment Arm (unadjusted)")
    fig.tight_layout()

    fig.savefig(KM_CURVES_OUTPUT_PATH, dpi=150)
    plt.close(fig)
    logging.info(f"Saved: {KM_CURVES_OUTPUT_PATH}")


def simulate_negative_control_outcome(data: pd.DataFrame) -> pd.DataFrame:
    """ Exponential hazard simulation approach with the treatment log-HR fixed at 0 """
    standardized = standardize_covariates(data)

    def hazard(log_lambda_0):
        linear_predictor = pd.Series(log_lambda_0, index=standardized.index)
        linear_predictor = linear_predictor + NEGATIVE_CONTROL_TRUE_LOG_HR * (data["treatment"] == "SGLT2i").astype(float)
        for col, coef in OUTCOME_COEFFICIENTS.items():
            linear_predictor = linear_predictor + coef * standardized[col]
        return np.exp(linear_predictor)

    low, high = -20.0, 5.0
    for _ in range(50):
        mid = (low + high) / 2
        cumulative_incidence = 1 - np.exp(-hazard(mid) * FOLLOWUP_DAYS)
        if cumulative_incidence.mean() < TARGET_OVERALL_EVENT_RATE:
            low = mid
        else:
            high = mid
    log_lambda_0 = (low + high) / 2

    final_hazard = hazard(log_lambda_0)

    rng = np.random.default_rng(RANDOM_SEED + 2)
    uniform_draw = rng.uniform(size=len(data))
    simulated_event_days = -np.log(uniform_draw) / final_hazard

    result = data[["patient_id", "treatment"]].copy()
    result["nc_event_days"] = np.minimum(simulated_event_days, FOLLOWUP_DAYS)
    result["nc_event_occurred"] = simulated_event_days <= FOLLOWUP_DAYS

    n_events = result["nc_event_occurred"].sum()
    logging.info(f"  Negative control: {n_events:,}/{len(result):,} patients ({n_events/len(result):.1%}) experienced the simulated event (true HR = 1.0 by design)")

    return result


def run_negative_control_check(data: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """ Fits naive and IPTW-weighted Cox models on the negative control outcome """
    logging.info("Simulating negative control outcome (same confounding structure, TRUE HR = 1.0) ...")
    nc_outcome = simulate_negative_control_outcome(data)
    merged = data.merge(nc_outcome[["patient_id", "nc_event_days", "nc_event_occurred"]], on="patient_id")

    results = []
    for label, use_weights in [("Naive", False), ("IPTW-weighted", True)]:
        cox_data = merged[["nc_event_days", "nc_event_occurred", "treatment"]].copy()
        cox_data["sglt2i"] = (cox_data["treatment"] == "SGLT2i").astype(int)
        cox_data = cox_data.drop(columns="treatment")

        cph = CoxPHFitter()
        if use_weights:
            cox_data["weight"] = weights.values
            cph.fit(cox_data, duration_col="nc_event_days", event_col="nc_event_occurred", weights_col="weight", robust=True)
        else:
            cph.fit(cox_data, duration_col="nc_event_days", event_col="nc_event_occurred")

        hr = np.exp(cph.params_["sglt2i"])
        ci_low, ci_high = np.exp(cph.confidence_intervals_.loc["sglt2i"])
        contains_null = ci_low <= 1.0 <= ci_high

        logging.info(f"  {label} negative control HR: {hr:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f}) - CI contains 1.0? {contains_null}")
        results.append({"model": label, "hr": hr, "ci_low": ci_low, "ci_high": ci_high, "contains_null": contains_null})

    table = pd.DataFrame(results)
    table.to_csv(NEGATIVE_CONTROL_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {NEGATIVE_CONTROL_OUTPUT_PATH}")

    if not table.loc[table["model"] == "IPTW-weighted", "contains_null"].iloc[0]:
        logging.info("  NOTE: adjusted negative control CI excludes 1.0 - suggests residual confounding the propensity model isn't fully capturing.")
    return table


def compute_e_value(estimate: float) -> float:
    """ 
    Compute the minimum strength of association that an unmeasured confounder would need to 
    have with both treatment and outcome to fully explain away the observed association, 
    given the covariates already adjusted for
    """
    ratio = estimate if estimate >= 1 else 1 / estimate
    return ratio + np.sqrt(ratio * (ratio - 1))


def run_e_value_analysis(doubly_robust_result: dict) -> None:
    """ Computes the E-value for the doubly-robust point estimate and for the confidence limit closest to the null """
    hr = doubly_robust_result["hr"]
    ci_low = doubly_robust_result["ci_low"]
    ci_high = doubly_robust_result["ci_high"]

    ci_limit_closest_to_null = ci_high if hr < 1 else ci_low

    e_value_point = compute_e_value(hr)
    e_value_ci = compute_e_value(ci_limit_closest_to_null)

    logging.info("E-value analysis (doubly-robust estimate):")
    logging.info(f"  Point estimate HR={hr:.3f} -> E-value = {e_value_point:.2f}")
    logging.info(f"  CI limit closest to null ({ci_limit_closest_to_null:.3f}) -> E-value = {e_value_ci:.2f}")


def run_propensity_analysis():
    data = load_analysis_dataset()
    log_separator()

    propensity_scores = estimate_propensity_scores(data)
    data["propensity_score"] = propensity_scores.values

    output = data[["patient_id", "treatment", "propensity_score"]].copy()
    output.to_csv(PROPENSITY_SCORES_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {PROPENSITY_SCORES_OUTPUT_PATH}")
    log_separator()

    logging.info("Computing stabilized IPTW weights ...")
    weights = compute_iptw_weights(propensity_scores, data["treatment"])

    unadjusted_smds = check_covariate_balance_by_treatment_arm(
        data[["patient_id"] + COVARIATE_COLUMNS], data[["patient_id", "treatment"]]
    )
    weighted_smds = check_weighted_covariate_balance(data, weights)

    naive_result = check_naive_treatment_effect(data[["hf_event_days", "hf_event_occurred", "treatment"]])
    iptw_result = fit_weighted_cox_model(data, weights)
    dr_result = fit_doubly_robust_cox_model(data, weights)

    build_results_table(naive_result, iptw_result, dr_result)
    plot_love_plot(unadjusted_smds, weighted_smds)
    plot_km_curves(data)

    log_separator()
    run_negative_control_check(data, weights)
    run_e_value_analysis(dr_result)