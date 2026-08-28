import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import contextmanager
from lifelines import CoxPHFitter

from . import injection
from . import analysis
from .injection import standardize_covariates
from .config import (
    MONTE_CARLO_RESULTS_PATH,
    MONTE_CARLO_PLOT_PATH,
    N_SEEDS,
    BASE_SEED,
    PATHOLOGICAL_HR_BOUNDS,
    RANDOM_SEED as DEFAULT_RANDOM_SEED,
)
from .helpers import log_separator, find_latest_completed_run


@contextmanager
def quiet_logging():
    """ Suppresses INFO-level logs """
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root_logger.setLevel(original_level)


def run_single_realization(cohort: pd.DataFrame, seed: int) -> dict:
    """ One full injection + analysis realization for a given seed """
    injection.RANDOM_SEED = seed

    with quiet_logging():
        assignment = injection.assign_treatment(cohort)
        outcome = injection.simulate_hf_outcome(cohort, assignment)

        data = cohort.merge(assignment[["patient_id", "treatment"]], on="patient_id") \
                     .merge(outcome[["patient_id", "hf_event_days", "hf_event_occurred"]], on="patient_id")

        propensity_scores = analysis.estimate_propensity_scores(data)
        weights = analysis.compute_iptw_weights(propensity_scores, data["treatment"])

        try:
            naive_result = injection.check_naive_treatment_effect(data[["hf_event_days", "hf_event_occurred", "treatment"]])
            iptw_result = analysis.fit_weighted_cox_model(data, weights)
            dr_result = analysis.fit_doubly_robust_cox_model(data, weights)
        except Exception as e:
            raise RuntimeError(
                f"ps_min={propensity_scores.min():.4f}, ps_max={propensity_scores.max():.4f}, "
                f"weight_max={weights.max():.2f}: {type(e).__name__}: {e}"
            ) from e

    row = {
        "seed": seed,
        "ps_min": propensity_scores.min(),
        "ps_max": propensity_scores.max(),
        "weight_max": weights.max(),
    }
    for prefix, result in [("naive", naive_result), ("iptw", iptw_result), ("dr", dr_result)]:
        row[f"{prefix}_hr"] = result["hr"]
        row[f"{prefix}_ci_low"] = result["ci_low"]
        row[f"{prefix}_ci_high"] = result["ci_high"]
        row[f"{prefix}_covers_truth"] = result["ci_low"] <= injection.TRUE_SGLT2I_HAZARD_RATIO <= result["ci_high"]

    return row


def summarize_monte_carlo_results(results: pd.DataFrame) -> None:
    """ Logs mean HR, std, and empirical CI coverage per estimator across all seeds """
    true_hr = injection.TRUE_SGLT2I_HAZARD_RATIO
    logging.info(f"Monte Carlo summary across {len(results)} seeds (true HR = {true_hr:.3f}):")
    for prefix, label in [("naive", "Naive"), ("iptw", "IPTW-weighted"), ("dr", "Doubly-robust")]:
        mean_hr = results[f"{prefix}_hr"].mean()
        std_hr = results[f"{prefix}_hr"].std()
        coverage_rate = results[f"{prefix}_covers_truth"].mean()
        logging.info(
            f"  {label}: mean HR={mean_hr:.3f} (std={std_hr:.3f}), "
            f"empirical CI coverage rate={coverage_rate:.1%} (ideal: ~95%)"
        )


def plot_hr_distribution(results: pd.DataFrame) -> None:
    """ Plots the distribution of HR estimates across seeds for each estimator against the true HR """
    true_hr = injection.TRUE_SGLT2I_HAZARD_RATIO

    fig, ax = plt.subplots(figsize=(7, 5))
    for prefix, label, color in [
        ("naive", "Naive", "tab:red"),
        ("iptw", "IPTW-weighted", "tab:blue"),
        ("dr", "Doubly-robust", "tab:green"),
    ]:
        ax.hist(results[f"{prefix}_hr"], bins=15, alpha=0.5, label=label, color=color)

    ax.axvline(true_hr, color="black", linestyle="--", linewidth=1.2, label=f"True HR = {true_hr:.2f}")
    ax.set_xlabel("Estimated Hazard Ratio (SGLT2i vs. DPP-4i)")
    ax.set_ylabel("Count across seeds")
    ax.set_title(f"Distribution of HR Estimates Across {len(results)} Monte Carlo Seeds")
    ax.legend()
    fig.tight_layout()

    fig.savefig(MONTE_CARLO_PLOT_PATH, dpi=150)
    plt.close(fig)
    logging.info(f"Saved: {MONTE_CARLO_PLOT_PATH}")


def find_pathological_seeds(results: pd.DataFrame) -> list:
    """ Flags seeds whose doubly-robust HR is implausible """
    low, high = PATHOLOGICAL_HR_BOUNDS
    pathological = results.loc[(results["dr_hr"] < low) | (results["dr_hr"] > high)]
    seeds = pathological["seed"].tolist()
    logging.info(f"Found {len(seeds)} pathological seed(s) (DR HR outside {PATHOLOGICAL_HR_BOUNDS}): {seeds}")

    return seeds


def investigate_pathological_seed(cohort: pd.DataFrame, seed: int) -> None:
    """ Checks quasi-separation among event cases per covariate, and doubly-robust coefficient stability """
    injection.RANDOM_SEED = seed

    with quiet_logging():
        assignment = injection.assign_treatment(cohort)
        outcome = injection.simulate_hf_outcome(cohort, assignment)
        data = cohort.merge(assignment[["patient_id", "treatment"]], on="patient_id") \
                     .merge(outcome[["patient_id", "hf_event_days", "hf_event_occurred"]], on="patient_id")

        propensity_scores = analysis.estimate_propensity_scores(data)
        weights = analysis.compute_iptw_weights(propensity_scores, data["treatment"])
        standardized = standardize_covariates(data)

    n_events = int(data["hf_event_occurred"].sum())
    logging.info(
        f"Seed {seed}: ps=[{propensity_scores.min():.3f}, {propensity_scores.max():.3f}], "
        f"weight_max={weights.max():.2f}, events={n_events}"
    )

    # Quasi-separation check
    events_only = data.loc[data["hf_event_occurred"]]
    separated_covariates = []
    if len(events_only) > 5:
        standardized_events = standardized.loc[events_only.index]
        for col in analysis.COVARIATE_COLUMNS:
            sglt2i_vals = standardized_events.loc[events_only["treatment"] == "SGLT2i", col]
            dpp4i_vals = standardized_events.loc[events_only["treatment"] == "DPP4i", col]
            if len(sglt2i_vals) > 0 and len(dpp4i_vals) > 0:
                if max(sglt2i_vals.min(), dpp4i_vals.min()) > min(sglt2i_vals.max(), dpp4i_vals.max()):
                    separated_covariates.append(col)

    if separated_covariates:
        logging.warning(f"  Quasi-separation among event cases for: {separated_covariates}")
    else:
        logging.info("  No quasi-separation among event cases for any covariate.")

    # Doubly-robust coefficient stability check
    cox_data = data[["hf_event_days", "hf_event_occurred", "treatment"]].copy()
    cox_data["sglt2i"] = (cox_data["treatment"] == "SGLT2i").astype(int)
    cox_data = cox_data.drop(columns="treatment")
    cox_data["weight"] = weights.values
    for col in analysis.COVARIATE_COLUMNS:
        cox_data[col] = standardized[col].values

    with quiet_logging():
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col="hf_event_days", event_col="hf_event_occurred", weights_col="weight", robust=True)

    unstable = [
        f"{name} (SE={se:.2f})"
        for name, se in zip(cph.params_.index, cph.standard_errors_)
        if se > 2
    ]

    if unstable:
        logging.warning(f"  Unstable DR coefficients (SE > 2): {unstable}")
    else:
        logging.info("  All DR coefficient SEs within normal range.")


def run_pathological_seed_diagnostics(cohort: pd.DataFrame, results: pd.DataFrame) -> None:
    """ Finds pathological seeds and runs stability diagnostics on each """
    seeds = find_pathological_seeds(results)
    for seed in seeds:
        investigate_pathological_seed(cohort, seed)


def run_monte_carlo_validation(n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
    """ Runs the full Monte Carlo validation: repeated injection+analysis realizations, summary, and diagnostics """
    source_run_dir = find_latest_completed_run("complete_case_cohort.csv")
    cohort_path = source_run_dir / "01_preprocessed_data" / "complete_case_cohort.csv"

    logging.info(f"Loading complete-case cohort from {cohort_path} ...")
    cohort = pd.read_csv(cohort_path, parse_dates=["metformin_start_date"])
    logging.info(f"  {len(cohort):,} patients loaded")
    log_separator()

    logging.info(f"Running {n_seeds} Monte Carlo realizations (seeds {base_seed}-{base_seed + n_seeds - 1}) ...")

    rows = []
    failed_seeds = []
    for i, seed in enumerate(range(base_seed, base_seed + n_seeds)):
        try:
            rows.append(run_single_realization(cohort, seed))
        except Exception as e:
            logging.warning(f"  Seed {seed} failed: {e}")
            failed_seeds.append(seed)

        if (i + 1) % 10 == 0:
            logging.info(f"  Completed {i + 1}/{n_seeds} realizations ...")

    if failed_seeds:
        logging.warning(f"{len(failed_seeds)}/{n_seeds} realizations failed to converge (seeds: {failed_seeds}) ")

    injection.RANDOM_SEED = DEFAULT_RANDOM_SEED

    results = pd.DataFrame(rows)
    results.to_csv(MONTE_CARLO_RESULTS_PATH, index=False)
    logging.info(f"Saved: {MONTE_CARLO_RESULTS_PATH} ({len(results)}/{n_seeds} successful)")
    log_separator()

    summarize_monte_carlo_results(results)
    plot_hr_distribution(results)
    log_separator()
    run_pathological_seed_diagnostics(cohort, results)