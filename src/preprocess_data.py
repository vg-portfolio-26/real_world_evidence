import logging
import pandas as pd
from pathlib import Path

from .config import (
    CONDITIONS_PATH,
    PATIENTS_PATH,
    MEDICATIONS_PATH,
    OBSERVATIONS_PATH,
    T2DM_PATIENTS_OUTPUT_PATH,
    METFORMIN_COHORT_OUTPUT_PATH,
    NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH,
    BASELINE_COVARIATES_OUTPUT_PATH,
    COMPLETE_CASE_COHORT_PATH,
    T2DM_INCLUSION_CODES,
    METFORMIN_CODE,
    ANTIDIABETIC_CODES,
    HF_INCLUSION_CODES,
    COVARIATE_OBSERVATION_CODES,
    CANONICAL_VARIANT_FILTER,
    BASELINE_WINDOW_DAYS_AFTER_INDEX,
    PLAUSIBLE_RANGES,
    SCALE_CORRECTION_CODES,
)
from .helpers import log_separator


def _strip_timezone(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """ Standardize dates to timezone-naive (UTC) for internal consistency """
    for col in date_columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df


def load_conditions(path: Path) -> pd.DataFrame:
    """ Loads condition records with timezone-naive dates """
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )
    return _strip_timezone(df, ["START", "STOP"])


def load_patients(path: Path) -> pd.DataFrame:
    """ Loads patient demographics with timezone-naive dates """
    df = pd.read_csv(
        path,
        usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE"],
        parse_dates=["BIRTHDATE", "DEATHDATE"],
    )
    df = df.rename(columns={"Id": "patient_id"})
    return _strip_timezone(df, ["BIRTHDATE", "DEATHDATE"])


def load_medications(path: Path) -> pd.DataFrame:
    """ Loads medication records with timezone-naive dates """
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )
    return _strip_timezone(df, ["START", "STOP"])


def load_baseline_observations(path: Path, patient_ids: set, chunksize: int = 1_000_000) -> pd.DataFrame:
    """ Reads in chunks, filtering each chunk to only our cohort's patients and only the 6 covariate CODEs needed due to memory constraints """
    matched_chunks = []
    total_rows_seen = 0
 
    for chunk in pd.read_csv(
        path,
        usecols=["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
        parse_dates=["DATE"],
        chunksize=chunksize,
    ):
        total_rows_seen += len(chunk)
        relevant = chunk.loc[
            chunk["PATIENT"].isin(patient_ids)
            & chunk["CODE"].isin(COVARIATE_OBSERVATION_CODES)
        ]
        matched_chunks.append(relevant)
 
    logging.info(f"  Scanned {total_rows_seen:,} total observation rows across all patients")
    observations = pd.concat(matched_chunks, ignore_index=True)
    return _strip_timezone(observations, ["DATE"])


def identify_t2dm_patients(conditions: pd.DataFrame) -> pd.DataFrame:
    """ Finds each patient's earliest qualifying T2DM diagnosis date across base and complication codes """
    mask = conditions["CODE"].astype(str).isin(T2DM_INCLUSION_CODES)
    t2dm_conditions = conditions.loc[mask].copy()

    matched_descriptions = t2dm_conditions["DESCRIPTION"].unique()
    logging.info(f"CODE-based match found {len(matched_descriptions)} distinct DESCRIPTION values:")
    for desc in matched_descriptions:
        logging.info(f"  - {desc}")

    # Collapse to one row per patient using their earliest diagnosis date
    # across ALL qualifying codes (base diagnosis or any complication)
    t2dm_conditions = t2dm_conditions.sort_values("START")
    first_diagnosis = (
        t2dm_conditions.groupby("PATIENT", as_index=False)
        .first()[["PATIENT", "START"]]
        .rename(columns={"PATIENT": "patient_id", "START": "t2dm_diagnosis_date"})
    )
    return first_diagnosis


def build_t2dm_cohort():
    """ Builds and saves the T2DM patient cohort from conditions and patient demographics """
    logging.info(f"Loading conditions from {CONDITIONS_PATH} ...")
    conditions = load_conditions(CONDITIONS_PATH)
    logging.info(f"  {len(conditions):,} condition records loaded")

    logging.info(f"Loading patients from {PATIENTS_PATH} ...")
    patients = load_patients(PATIENTS_PATH)
    logging.info(f"  {len(patients):,} patients loaded")
    log_separator()

    logging.info("Identifying T2DM patients ...")
    t2dm_patients = identify_t2dm_patients(conditions)
    logging.info(f"  {len(t2dm_patients):,} unique patients with a T2DM diagnosis")

    # Attach baseline patient attributes
    t2dm_patients = t2dm_patients.merge(patients, on="patient_id", how="left")

    t2dm_patients.to_csv(T2DM_PATIENTS_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {T2DM_PATIENTS_OUTPUT_PATH}")


def identify_metformin_new_users(medications: pd.DataFrame, t2dm_patient_ids: set) -> pd.DataFrame:
    """ Finds T2DM patients whose first antidiabetic drug was metformin """
    antidiabetic_meds = medications.loc[
        medications["PATIENT"].isin(t2dm_patient_ids)
        & medications["CODE"].astype(str).isin(ANTIDIABETIC_CODES)
    ].copy()
 
    logging.info(f"  {len(antidiabetic_meds):,} antidiabetic medication records found for T2DM-cohort patients")
 
    metformin_meds = antidiabetic_meds.loc[antidiabetic_meds["CODE"].astype(str) == METFORMIN_CODE]
    metformin_start = metformin_meds.groupby("PATIENT")["START"].min().rename("metformin_start_date")
    logging.info(f"  {len(metformin_start):,} T2DM patients have at least one metformin record")
 
    other_meds = antidiabetic_meds.loc[antidiabetic_meds["CODE"].astype(str) != METFORMIN_CODE]
    other_start = other_meds.groupby("PATIENT")["START"].min().rename("other_antidiabetic_start_date")
 
    cohort = metformin_start.to_frame().join(other_start, how="left")
 
    is_new_user = (
        cohort["other_antidiabetic_start_date"].isna()
        | (cohort["metformin_start_date"] <= cohort["other_antidiabetic_start_date"])
    )
    n_excluded = (~is_new_user).sum()
    logging.info(f"  {n_excluded:,} patients excluded: had another antidiabetic drug before their first metformin record")
 
    metformin_cohort = cohort.loc[is_new_user].reset_index().rename(columns={"PATIENT": "patient_id"})

    return metformin_cohort[["patient_id", "metformin_start_date"]]
 
 
def build_metformin_cohort():
    """ Builds and saves the metformin new-user cohort from the T2DM cohort and medication records """
    logging.info(f"Loading T2DM cohort from {T2DM_PATIENTS_OUTPUT_PATH} ...")
    t2dm_patients = pd.read_csv(T2DM_PATIENTS_OUTPUT_PATH, usecols=["patient_id"])
    t2dm_patient_ids = set(t2dm_patients["patient_id"].unique())
    logging.info(f"  {len(t2dm_patient_ids):,} T2DM patients loaded")
 
    logging.info(f"Loading medications from {MEDICATIONS_PATH} ...")
    medications = load_medications(MEDICATIONS_PATH)
    logging.info(f"  {len(medications):,} medication records loaded")
    log_separator()
 
    logging.info("Identifying metformin new-users (monotherapy at treatment start) ...")
    metformin_cohort = identify_metformin_new_users(medications, t2dm_patient_ids)
    logging.info(f"  {len(metformin_cohort):,} patients qualify as metformin new-users")
 
    metformin_cohort.to_csv(METFORMIN_COHORT_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {METFORMIN_COHORT_OUTPUT_PATH}")


def identify_no_prior_hf_patients(conditions: pd.DataFrame, metformin_cohort: pd.DataFrame) -> pd.DataFrame:
    """ Excludes any metformin-cohort patient with an heart failure diagnosis on or before their metformin_start_date """
    hf_conditions = conditions.loc[conditions["CODE"].astype(str).isin(HF_INCLUSION_CODES)].copy()
 
    earliest_hf_date = hf_conditions.groupby("PATIENT")["START"].min().rename("hf_diagnosis_date")
 
    cohort = metformin_cohort.set_index("patient_id").join(earliest_hf_date, how="left")
 
    has_prior_hf = cohort["hf_diagnosis_date"].notna() & (
        cohort["hf_diagnosis_date"] <= cohort["metformin_start_date"]
    )
    n_excluded = has_prior_hf.sum()
    logging.info(f"  {n_excluded:,} metformin-cohort patients excluded: heart failure diagnosis on or before their metformin_start_date")
 
    no_prior_hf_cohort = cohort.loc[~has_prior_hf].reset_index()

    return no_prior_hf_cohort[["patient_id", "metformin_start_date"]]


def build_no_prior_hf_cohort():
    """ Builds and saves the metformin cohort restricted to patients with no prior heart failure """
    logging.info(f"Loading conditions from {CONDITIONS_PATH} ...")
    conditions = load_conditions(CONDITIONS_PATH)
    logging.info(f"  {len(conditions):,} condition records loaded")
 
    logging.info(f"Loading metformin cohort from {METFORMIN_COHORT_OUTPUT_PATH} ...")
    metformin_cohort = pd.read_csv(METFORMIN_COHORT_OUTPUT_PATH, parse_dates=["metformin_start_date"])
    logging.info(f"  {len(metformin_cohort):,} metformin-cohort patients loaded")
    log_separator()
 
    logging.info("Identifying patients with no prior heart failure diagnosis at baseline ...")
    no_prior_hf_cohort = identify_no_prior_hf_patients(conditions, metformin_cohort)
    logging.info(f"  {len(no_prior_hf_cohort):,} patients qualify (no prior heart failure)")
 
    no_prior_hf_cohort.to_csv(NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH}")
 
 
def extract_baseline_covariates(observations: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """ For each patient and each covariate CODE, take the value closest to that patient's metformin_start_date, among observations within the baseline window """
    for code, (description, units) in CANONICAL_VARIANT_FILTER.items():
        bad_variant_mask = (
            (observations["CODE"] == code)
            & ~((observations["DESCRIPTION"] == description) & (observations["UNITS"] == units))
        )
        n_dropped = bad_variant_mask.sum()
        if n_dropped > 0:
            logging.info(f"  Dropping {n_dropped:,} non-canonical-variant records for CODE {code}")
        observations = observations.loc[~bad_variant_mask]
 
    observations = observations.merge(
        cohort[["patient_id", "metformin_start_date"]],
        left_on="PATIENT", right_on="patient_id", how="inner",
    )
 
    observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")

    # Apply scale correction 
    for code, factor in SCALE_CORRECTION_CODES.items():
        code_mask = observations["CODE"] == code
        name = COVARIATE_OBSERVATION_CODES[code]
        low, high = PLAUSIBLE_RANGES[name]
 
        # Only rescale values that are implausible on their original scale and would become plausible after correction
        originally_implausible = code_mask & ~observations["VALUE"].between(low, high)
        would_become_plausible = (observations["VALUE"] * factor).between(low, high)
        to_correct = originally_implausible & would_become_plausible
 
        n_corrected = to_correct.sum()
        if n_corrected > 0:
            logging.info(
                f"  {name} (CODE {code}): {n_corrected:,} records rescaled (x{factor})"
            )
            observations.loc[to_correct, "VALUE"] = observations.loc[to_correct, "VALUE"] * factor

    # Drop implausible covariate values
    covariate_names_by_code = COVARIATE_OBSERVATION_CODES
    for code, name in covariate_names_by_code.items():
        low, high = PLAUSIBLE_RANGES[name]
        code_mask = observations["CODE"] == code
        implausible_mask = code_mask & ~observations["VALUE"].between(low, high)
        n_implausible = implausible_mask.sum()
        if n_implausible > 0:
            logging.info(
                f"  {name} (CODE {code}): {n_implausible:,} observation records outside "
                f"plausible range [{low}, {high}] treated as missing"
            )
        observations = observations.loc[~implausible_mask]
 
    window_upper_bound = observations["metformin_start_date"] + pd.Timedelta(days=BASELINE_WINDOW_DAYS_AFTER_INDEX)
    baseline_eligible = observations.loc[observations["DATE"] <= window_upper_bound].copy()
 
    baseline_eligible["days_from_index"] = (
        baseline_eligible["DATE"] - baseline_eligible["metformin_start_date"]
    ).dt.total_seconds() / 86400
    baseline_eligible["abs_days_from_index"] = baseline_eligible["days_from_index"].abs()
 
    baseline_eligible = baseline_eligible.sort_values("abs_days_from_index")
    closest_per_patient_code = (
        baseline_eligible.groupby(["patient_id", "CODE"], as_index=False)
        .first()[["patient_id", "CODE", "VALUE"]]
    )
 
    # Pivot to one row per patient, one column per covariate
    wide = closest_per_patient_code.pivot(index="patient_id", columns="CODE", values="VALUE")
    wide = wide.rename(columns=COVARIATE_OBSERVATION_CODES).reset_index()
 
    for code, name in COVARIATE_OBSERVATION_CODES.items():
        if name not in wide.columns:
            wide[name] = pd.NA
        n_missing = wide[name].isna().sum()
        logging.info(f"  {name} (CODE {code}): {len(wide) - n_missing:,} patients have a baseline value, {n_missing:,} missing")
 
    return wide


def build_baseline_covariates():
    """ Loads the no-prior-HF cohort and observations, extracts baseline covariates, and saves the result """
    logging.info(f"Baseline window: DATE <= metformin_start_date + {BASELINE_WINDOW_DAYS_AFTER_INDEX} days")
    logging.info(f"Loading no-prior-heart-failure cohort from {NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH} ...")
    cohort = pd.read_csv(NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH, parse_dates=["metformin_start_date"])
    cohort = _strip_timezone(cohort, ["metformin_start_date"])
    patient_ids = set(cohort["patient_id"].unique())
    logging.info(f"  {len(patient_ids):,} cohort patients loaded")
 
    logging.info(f"Loading observations from {OBSERVATIONS_PATH} ...")
    observations = load_baseline_observations(OBSERVATIONS_PATH, patient_ids)
    logging.info(f"  {len(observations):,} observation records loaded")
    log_separator()
 
    logging.info("Extracting baseline covariates ...")
    covariates = extract_baseline_covariates(observations, cohort)
 
    result = cohort.merge(covariates, on="patient_id", how="left")
 
    result.to_csv(BASELINE_COVARIATES_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {BASELINE_COVARIATES_OUTPUT_PATH}")


def build_complete_case_cohort():
    """ Filters baseline_covariates.csv down to complete cases, and attaches sex, race, age at metformin_start_date """
    logging.info(f"Loading baseline covariates from {BASELINE_COVARIATES_OUTPUT_PATH} ...")
    covariates = pd.read_csv(BASELINE_COVARIATES_OUTPUT_PATH, parse_dates=["metformin_start_date"])
    covariates = _strip_timezone(covariates, ["metformin_start_date"])
    logging.info(f"  {len(covariates):,} cohort patients loaded")
 
    logging.info(f"Loading patients from {PATIENTS_PATH} ...")
    patients = load_patients(PATIENTS_PATH)
    patients = patients[["patient_id", "BIRTHDATE", "GENDER", "RACE"]]
    logging.info(f"  {len(patients):,} patients loaded")

    log_separator()
    logging.info("Building complete-case cohort ...")
    merged = covariates.merge(patients, on="patient_id", how="left")
    merged["age"] = (merged["metformin_start_date"] - merged["BIRTHDATE"]).dt.days / 365.25
 
    covariate_columns = list(COVARIATE_OBSERVATION_CODES.values())
    complete_case = merged.dropna(subset=covariate_columns).copy()
    logging.info(
        f"  Complete-case cohort (all {len(covariate_columns)} covariates present): "
        f"{len(complete_case):,}/{len(merged):,} patients"
    )
 
    output_columns = ["patient_id", "metformin_start_date", "age", "GENDER", "RACE"] + covariate_columns
    complete_case = complete_case[output_columns]
 
    complete_case.to_csv(COMPLETE_CASE_COHORT_PATH, index=False)
    logging.info(f"Saved: {COMPLETE_CASE_COHORT_PATH}")
