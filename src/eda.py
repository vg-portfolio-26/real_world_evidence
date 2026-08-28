import logging
import pandas as pd
from pathlib import Path

from .config import (
    CONDITIONS_PATH,
    MEDICATIONS_PATH,
    OBSERVATIONS_PATH,
    PATIENTS_PATH,
    T2DM_PATIENTS_OUTPUT_PATH,
    HF_INCLUSION_CODES,
    METFORMIN_COHORT_OUTPUT_PATH,
    BASELINE_WINDOW_DAYS_AFTER_INDEX,
    BASELINE_COVARIATES_OUTPUT_PATH,
    NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH,
    PLAUSIBLE_RANGES,
    COVARIATE_OBSERVATION_CODES,
    EXPLORATORY_DIABETES_PATTERN,
    BASE_DX_DESCRIPTION,
    COMPLICATION_PATTERN,
    EXPLORATORY_HF_PATTERN,
    T2DM_EXCLUDED_CODES,
    ANTIDIABETIC_PATTERN,
    BASELINE_COVARIATE_PATTERN,
)
from .helpers import log_separator


def load_conditions(path: Path) -> pd.DataFrame:
    """ Loads condition records for exploratory analysis """
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )

    return df


def load_medications(path: Path) -> pd.DataFrame:
    """ Loads medication records for exploratory analysis """
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )

    return df


def load_observations_for_patients(path: Path, patient_ids: set, chunksize: int = 1_000_000) -> pd.DataFrame:
    """ Read in chunks and filter each chunk down to only our cohort's patients due to memory constraints """
    matched_chunks = []
    total_rows_seen = 0
 
    for chunk in pd.read_csv(
        path,
        usecols=["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
        parse_dates=["DATE"],
        chunksize=chunksize,
    ):
        total_rows_seen += len(chunk)
        matched_chunks.append(chunk.loc[chunk["PATIENT"].isin(patient_ids)])
 
    logging.info(f"  Scanned {total_rows_seen:,} observation rows across all patients")

    result = pd.concat(matched_chunks, ignore_index=True)
    if pd.api.types.is_datetime64tz_dtype(result["DATE"]):
        result["DATE"] = result["DATE"].dt.tz_localize(None)

    return result


def explore_diabetes_related_descriptions(conditions: pd.DataFrame) -> None:
    """
    Log every unique DESCRIPTION containing 'diabetes' (case-insensitive),
    with counts, sorted descending. This is the broad grep that revealed
    the base-diagnosis wording mismatch and the ambiguous/ excluded categories
    """
    mask = conditions["DESCRIPTION"].str.contains(
        EXPLORATORY_DIABETES_PATTERN, case=False, na=False
    )
    counts = (
        conditions.loc[mask, "DESCRIPTION"]
        .value_counts()
        .sort_values(ascending=False)
    )

    logging.info("All DESCRIPTION values containing 'diabetes' (record counts):")
    for description, count in counts.items():
        logging.info(f"  {count:>7,}  {description}")


def compare_base_dx_vs_complication_only(conditions: pd.DataFrame) -> None:
    """
    Quantify the overlap between:
      - patients with the explicit base T2DM diagnosis
      - patients with any T2DM-specific complication code
    to determine whether requiring the base diagnosis alone would wrongly exclude real T2DM patients
    """
    base_dx_patients = set(
        conditions.loc[
            conditions["DESCRIPTION"] == BASE_DX_DESCRIPTION, "PATIENT"
        ].unique()
    )

    complication_mask = conditions["DESCRIPTION"].str.contains(
        COMPLICATION_PATTERN, case=False, na=False
    )
    complication_patients = set(conditions.loc[complication_mask, "PATIENT"].unique())

    complication_only = complication_patients - base_dx_patients

    logging.info("Base diagnosis vs. complication-code overlap check:")
    logging.info(f"  Patients with base T2DM diagnosis:         {len(base_dx_patients):,}")
    logging.info(f"  Patients with any T2DM complication code:  {len(complication_patients):,}")
    logging.info(
        f"  Complication-only patients: {len(complication_only):,} "
        f"({len(complication_only) / len(complication_patients):.1%} of complication group)"
    )

    if len(complication_only) / max(len(complication_patients), 1) > 0.10:
        logging.info("The base diagnosis or any T2DM-specific complication code will be used as the inclusion rule")
    else:
        logging.info("The base diagnosis alone is a reasonable inclusion rule")


def log_excluded_categories(conditions: pd.DataFrame) -> None:
    """ Logs record/patient counts for codes deliberately excluded from the T2DM inclusion rule """
    logging.info("Excluded / ambiguous codes (not counted as T2DM evidence alone):")
    for code, reason in T2DM_EXCLUDED_CODES.items():
        subset = conditions.loc[conditions["CODE"].astype(str) == code]
        n_patients = subset["PATIENT"].nunique()
        logging.info(
            f"  CODE {code} ({reason}): {len(subset):,} records, {n_patients:,} unique patients"
        )


def check_code_vs_description_matching(conditions: pd.DataFrame, pattern: str, label: str) -> None:
    """ Investigate whether CODE is a more reliable identifier than free-text DESCRIPTION matching for T2DM-related conditions """
    logging.info(f"Checking CODE vs DESCRIPTION matching for {label}-related conditions ...")
    mask = conditions["DESCRIPTION"].str.contains(pattern, case=False, na=False)
    related = conditions.loc[mask, ["CODE", "DESCRIPTION"]]
 
    code_to_descriptions = related.groupby("CODE")["DESCRIPTION"].unique()
    description_to_codes = related.groupby("DESCRIPTION")["CODE"].unique()
 
    logging.info(
        f"  Checked {len(code_to_descriptions)} distinct CODEs against "
        f"  {len(description_to_codes)} distinct DESCRIPTIONs for {label}-related conditions"
    )
 
    multi_description_codes = code_to_descriptions[code_to_descriptions.apply(len) > 1]
    multi_code_descriptions = description_to_codes[description_to_codes.apply(len) > 1]
 
    if len(multi_description_codes) == 0 and len(multi_code_descriptions) == 0:
        logging.info("Every CODE maps to exactly one DESCRIPTION and vice versa; text and code are consistent 1:1")
    else:
        if len(multi_description_codes) > 0:
            logging.info("WARNING: CODE values mapped to more than one DESCRIPTION:")
            for code, descriptions in multi_description_codes.items():
                logging.info(f"  CODE {code}: {list(descriptions)}")
        if len(multi_code_descriptions) > 0:
            logging.info("WARNING: DESCRIPTION values mapped to more than one CODE:")
            for description, codes in multi_code_descriptions.items():
                logging.info(f"  '{description}': CODEs {list(codes)}")


def run_eda_conditions():
    """ Runs the full EDA pass supporting the T2DM cohort inclusion logic """
    logging.info(f"Loading conditions from {CONDITIONS_PATH} ...")
    conditions = load_conditions(CONDITIONS_PATH)
    logging.info(f"  {len(conditions):,} condition records loaded")
    log_separator()

    explore_diabetes_related_descriptions(conditions)
    log_separator()

    compare_base_dx_vs_complication_only(conditions)
    log_separator()

    log_excluded_categories(conditions)
    log_separator()

    check_code_vs_description_matching(conditions, EXPLORATORY_DIABETES_PATTERN, "diabetes")


def explore_medications_for_t2dm_cohort(t2dm_patient_ids: set) -> None:
    """ Full audit of every medication prescribed to T2DM-cohort patients """
    logging.info(f"Loading medications from  {MEDICATIONS_PATH} ...")
    medications = load_medications(MEDICATIONS_PATH)
    logging.info(f"  {len(medications):,} medication records loaded")
 
    cohort_meds = medications.loc[medications["PATIENT"].isin(t2dm_patient_ids)]
    logging.info(f"  {len(cohort_meds):,} medication records belong to T2DM-cohort patients")
    log_separator()
 
    summary = (
        cohort_meds.groupby(["CODE", "DESCRIPTION"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
 
    logging.info(f"Top 15 medications by unique-patient count within the T2DM cohort (from a total of {len(summary)} distinct drugs):")
    for _, row in summary.head(15).iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records"
        )

    return cohort_meds


def check_for_missed_antidiabetic_drugs(cohort_meds: pd.DataFrame) -> None:
    """ Targeted check across all medications for T2DM-cohort patients, specifically searching for antidiabetic drug name patterns """
    logging.info(f"Antidiabetic-pattern check across all {len(cohort_meds):,} T2DM-cohort medication records:")
    mask = cohort_meds["DESCRIPTION"].str.contains(ANTIDIABETIC_PATTERN, case=False, na=False)
    matched = cohort_meds.loc[mask]
 
    summary = (
        matched.groupby(["CODE", "DESCRIPTION"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
 
    logging.info(f"  {len(summary)} distinct drugs matched:")
    for _, row in summary.iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records"
        )

 
def run_eda_medications():
    """ Runs the full EDA pass supporting the metformin new-user/monotherapy cohort logic """
    logging.info(f"Loading T2DM patient IDs from {T2DM_PATIENTS_OUTPUT_PATH} ...")
    t2dm_patients = pd.read_csv(T2DM_PATIENTS_OUTPUT_PATH, usecols=["patient_id"])
    t2dm_patient_ids = set(t2dm_patients["patient_id"].unique())
    logging.info(f"  {len(t2dm_patient_ids):,} patients IDs loaded")
 
    cohort_meds = explore_medications_for_t2dm_cohort(t2dm_patient_ids)
    log_separator()
    check_for_missed_antidiabetic_drugs(cohort_meds)


def explore_heart_failure_related_descriptions(conditions: pd.DataFrame) -> None:
    """ Discovery pass for heart failure related conditions """
    mask = conditions["DESCRIPTION"].str.contains(EXPLORATORY_HF_PATTERN, case=False, na=False)
    matched = conditions.loc[mask]
 
    summary = (
        matched.groupby(["CODE", "DESCRIPTION"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_records", ascending=False)
    )
 
    logging.info("All CODE/DESCRIPTION values containing 'heart failure':")
    for _, row in summary.iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} "
            f"- {row['n_records']:,} records, {row['n_patients']:,} unique patients"
        )
 
 
def compare_hf_patients_vs_metformin_cohort(conditions: pd.DataFrame, metformin_patient_ids: set) -> None:
    """ How many heart failure patients exist dataset-wide, and how many of those are also in the metformin cohort """
    logging.info("Checking heart failure patients share in both condition and metformin cohort table ...")
    hf_conditions = conditions.loc[conditions["CODE"].astype(str).isin(HF_INCLUSION_CODES)]
    hf_patient_ids = set(hf_conditions["PATIENT"].unique())
 
    logging.info(f"  {len(hf_conditions):,} heart failure condition records found, representing {len(hf_patient_ids):,} unique patients (dataset-wide)")
 
    overlap = hf_patient_ids & metformin_patient_ids
    logging.info(
        f"  {len(overlap):,} of those {len(hf_patient_ids):,} heart failure patients "
        f"are also in the {len(metformin_patient_ids):,} patient metformin cohort"
    )
 
 
def run_eda_heart_failure():
    """ EDA supporting the "no prior heart failure at baseline" cohort exclusion step """
    logging.info(f"Loading conditions from {CONDITIONS_PATH} ...")
    conditions = load_conditions(CONDITIONS_PATH)
    logging.info(f"  {len(conditions):,} condition records loaded")

    logging.info(f"Loading metformin-cohort patient IDs from {METFORMIN_COHORT_OUTPUT_PATH} ...")
    metformin_cohort = pd.read_csv(METFORMIN_COHORT_OUTPUT_PATH, usecols=["patient_id"])
    metformin_patient_ids = set(metformin_cohort["patient_id"].unique())
    logging.info(f"  {len(metformin_patient_ids):,} metformin-cohort patient IDs loaded")
    log_separator()
 
    explore_heart_failure_related_descriptions(conditions)
    check_code_vs_description_matching(conditions, EXPLORATORY_HF_PATTERN, "heart failure")
    compare_hf_patients_vs_metformin_cohort(conditions, metformin_patient_ids)


def explore_observations_for_cohort(patient_ids: set) -> pd.DataFrame:
    """ Discovery pass over observations.csv, restricted to our current cohort """
    logging.info(f"Loading observations from {OBSERVATIONS_PATH} ...")
    cohort_obs = load_observations_for_patients(OBSERVATIONS_PATH, patient_ids)
    logging.info(f"  {len(cohort_obs):,} observation records loaded")
    log_separator()
 
    summary = (
        cohort_obs.groupby(["CODE", "DESCRIPTION", "UNITS"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
 
    logging.info(f"Top 15 observations by unique-patient count within the cohort (from a total of {len(summary)} distinct observation types):")
    for _, row in summary.head(15).iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} ({row['UNITS']}) "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records"
        )
 
    return cohort_obs
 
 
def check_for_target_covariate_observations(cohort_obs: pd.DataFrame) -> None:
    """ Check across all observations for cohort patients for needed covariates """
    mask = cohort_obs["DESCRIPTION"].str.contains(BASELINE_COVARIATE_PATTERN, case=False, na=False)
    matched = cohort_obs.loc[mask]
 
    summary = (
        matched.groupby(["CODE", "DESCRIPTION", "UNITS"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
 
    logging.info(f"Target-covariate check across all {len(cohort_obs):,} cohort observation records:")
    logging.info(f"  {len(summary)} distinct matching observation types found:")
    for _, row in summary.iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} ({row['UNITS']}) "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records"
        )


def check_code_consistency_for_observations(cohort_obs: pd.DataFrame, codes: dict) -> set:
    """ Check every covariate CODE we plan to use and flag any with more than one DESCRIPTION/UNITS variant """
    logging.info(f"Checking DESCRIPTION/UNITS consistency for {len(codes)} target covariate CODEs:")
    flagged_codes = set()
 
    for code, label in codes.items():
        subset = cohort_obs.loc[cohort_obs["CODE"] == code]
        n_variants = subset[["DESCRIPTION", "UNITS"]].drop_duplicates().shape[0]
 
        if n_variants <= 1:
            logging.info(f"  CODE {code} ({label}): consistent, 1 DESCRIPTION/UNITS combination")
        else:
            logging.info(f"  CODE {code} ({label}): WARNING - {n_variants} distinct DESCRIPTION/UNITS combinations found")
            flagged_codes.add(code)
 
    return flagged_codes


def investigate_duplicate_codes(cohort_obs: pd.DataFrame, codes_to_investigate: set) -> None:
    """ Shows the date range each variant covers, to check whether it's a temporal migration or something else """
    logging.info(f"Investigating {len(codes_to_investigate)} duplicate-looking CODE(s): {codes_to_investigate}")
    subset = cohort_obs.loc[cohort_obs["CODE"].isin(codes_to_investigate)]
 
    summary = (
        subset.groupby(["CODE", "DESCRIPTION", "UNITS"])
        .agg(
            n_records=("PATIENT", "size"),
            n_patients=("PATIENT", "nunique"),
            earliest_date=("DATE", "min"),
            latest_date=("DATE", "max"),
        )
        .reset_index()
        .sort_values(["CODE", "earliest_date"])
    )
 
    for _, row in summary.iterrows():
        logging.info(
            f"  CODE {row['CODE']}: '{row['DESCRIPTION']}' ({row['UNITS']}) "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records, "
            f"date range {row['earliest_date'].date()} to {row['latest_date'].date()}"
        )


def check_value_distributions_for_duplicate_codes(cohort_obs: pd.DataFrame, codes_to_investigate: set) -> None:
    """ Check whether the VALUE distributions of each CODE/DESCRIPTION/UNITS variant for a given CODE look consistent with each other """
    logging.info(f"VALUE distribution by CODE/DESCRIPTION/UNITS variant for {codes_to_investigate}:")
    subset = cohort_obs.loc[cohort_obs["CODE"].isin(codes_to_investigate)].copy()
    subset["VALUE"] = pd.to_numeric(subset["VALUE"], errors="coerce")
 
    for (code, description, units), group in subset.groupby(["CODE", "DESCRIPTION", "UNITS"]):
        stats = group["VALUE"].describe()
        logging.info(
            f"  CODE {code}: '{description}' ({units}) - "
            f"n={int(stats['count']):,}, mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, 25%={stats['25%']:.2f}, median={stats['50%']:.2f}, "
            f"75%={stats['75%']:.2f}, max={stats['max']:.2f}"
        )


def analyze_nearest_observation_distance_distribution(cohort_obs: pd.DataFrame, codes: dict) -> None:
    """
    Computes, for all cohort patients, the distance in days to their nearest observation and logs:
      - a percentile breakdown, revealing the "elbow" where a same-visit cluster ends and a much later cluster begins
      - a single summary line showing how many patients the currently configured window captures
    """
    cohort = pd.read_csv(NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH, parse_dates=["metformin_start_date"])
    if pd.api.types.is_datetime64tz_dtype(cohort["metformin_start_date"]):
        cohort["metformin_start_date"] = cohort["metformin_start_date"].dt.tz_localize(None)
 
    merged = cohort_obs.merge(
        cohort[["patient_id", "metformin_start_date"]],
        left_on="PATIENT", right_on="patient_id", how="inner",
    )
    merged["days_from_index"] = (merged["DATE"] - merged["metformin_start_date"]).dt.total_seconds() / 86400
    merged["abs_days_from_index"] = merged["days_from_index"].abs()
 
    percentiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0]
    n_cohort = len(cohort)
    window_upper = BASELINE_WINDOW_DAYS_AFTER_INDEX
 
    logging.info(f"Nearest-observation distance (days) distribution across all {n_cohort:,} cohort patients (current window: DATE <= index + {window_upper} days):")
    for code, label in codes.items():
        code_obs = merged.loc[merged["CODE"] == code]
        nearest = code_obs.sort_values("abs_days_from_index").groupby("patient_id").first()
 
        quantiles = nearest["abs_days_from_index"].quantile(percentiles)
        formatted = ", ".join(f"p{int(p*100)}={v:.0f}" for p, v in quantiles.items())
 
        n_within_window = (code_obs.loc[code_obs["days_from_index"] <= window_upper, "patient_id"].nunique())
        logging.info(f"  CODE {code} ({label}): {formatted}  |  within current window: {n_within_window:,}/{n_cohort:,}")


def analyze_implausible_values(cohort_obs: pd.DataFrame, code: str, label: str) -> None:
    """ Logs the distribution of the implausible values """
    low, high = PLAUSIBLE_RANGES[label]
 
    code_obs = cohort_obs.loc[cohort_obs["CODE"] == code].copy()
    code_obs["VALUE"] = pd.to_numeric(code_obs["VALUE"], errors="coerce")
 
    implausible = code_obs.loc[~code_obs["VALUE"].between(low, high)]
    logging.info(
        f"Investigating {len(implausible):,} implausible {label} (CODE {code}) values "
        f"(plausible range: [{low}, {high}]):"
    )
 
    if len(implausible) == 0:
        logging.info("  No implausible values found.")
        return
 
    stats = implausible["VALUE"].describe()
    logging.info(
        f"  Value distribution: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
        f"min={stats['min']:.3f}, 25%={stats['25%']:.3f}, median={stats['50%']:.3f}, "
        f"75%={stats['75%']:.3f}, max={stats['max']:.3f}"
    )
 
    # Check specifically for a scale/unit mix-up
    scale_shifted = implausible["VALUE"] * 100
    n_recoverable_by_scaling = scale_shifted.between(low, high).sum()
    logging.info(
        f"  Of these, {n_recoverable_by_scaling:,} would fall within the plausible range if multiplied by 100"
    )

 
def run_eda_observations():
    """ Runs the full EDA pass supporting the baseline-covariate extraction logic """
    logging.info(f"Loading no-prior-heart-failure-cohort patient IDs from {NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH} ...")
    cohort = pd.read_csv(NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH, usecols=["patient_id"])
    patient_ids = set(cohort["patient_id"].unique())
    logging.info(f"  {len(patient_ids):,} patient IDs loaded")
 
    cohort_obs = explore_observations_for_cohort(patient_ids)
    log_separator()
    check_for_target_covariate_observations(cohort_obs)
    log_separator()

    flagged_codes = check_code_consistency_for_observations(cohort_obs, COVARIATE_OBSERVATION_CODES)

    if flagged_codes:
        log_separator()
        investigate_duplicate_codes(cohort_obs, flagged_codes)
        log_separator()
        check_value_distributions_for_duplicate_codes(cohort_obs, flagged_codes)
    else:
        logging.info("No target covariate CODEs flagged - all consistent, no further value-distribution check needed")

    log_separator()
    analyze_nearest_observation_distance_distribution(cohort_obs, COVARIATE_OBSERVATION_CODES)

    log_separator()
    for code, label in COVARIATE_OBSERVATION_CODES.items():
        analyze_implausible_values(cohort_obs, code, label)


def describe_complete_case_cohort_for_injection_calibration():
    """ Descriptive pass over the complete-case cohort """
    logging.info(f"Loading baseline covariates from {BASELINE_COVARIATES_OUTPUT_PATH} ...")
    covariates = pd.read_csv(BASELINE_COVARIATES_OUTPUT_PATH, parse_dates=["metformin_start_date"])
    if pd.api.types.is_datetime64tz_dtype(covariates["metformin_start_date"]):
        covariates["metformin_start_date"] = covariates["metformin_start_date"].dt.tz_localize(None)
    logging.info(f"  {len(covariates):,} cohort patients loaded")
 
    logging.info(f"Loading patients from {PATIENTS_PATH} ...")
    patients = pd.read_csv(PATIENTS_PATH, usecols=["Id", "BIRTHDATE"], parse_dates=["BIRTHDATE"])
    patients = patients.rename(columns={"Id": "patient_id"})
    if pd.api.types.is_datetime64tz_dtype(patients["BIRTHDATE"]):
        patients["BIRTHDATE"] = patients["BIRTHDATE"].dt.tz_localize(None)
    
    merged = covariates.merge(patients, on="patient_id", how="left")
    merged["age"] = (merged["metformin_start_date"] - merged["BIRTHDATE"]).dt.days / 365.25
 
    covariate_columns = list(COVARIATE_OBSERVATION_CODES.values())
    complete_case = merged.dropna(subset=covariate_columns)
    logging.info(
        f"  Complete-case cohort (all {len(covariate_columns)} covariates present): "
        f"{len(complete_case):,}/{len(merged):,} patients"
    )
    
    log_separator()
    logging.info("Descriptive statistics for complete-case cohort:")
    for column in ["age"] + covariate_columns:
        stats = complete_case[column].describe()
        logging.info(
            f"  {column}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
            f"min={stats['min']:.1f}, 25%={stats['25%']:.1f}, median={stats['50%']:.1f}, "
            f"75%={stats['75%']:.1f}, max={stats['max']:.1f}"
        )
