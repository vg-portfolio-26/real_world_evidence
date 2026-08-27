import logging
import pandas as pd
from pathlib import Path

from .helpers import PREPROCESSED_DATA_DIR, log_separator

RAW_DATA_DIR = Path("raw_data/csv")

CONDITIONS_PATH = RAW_DATA_DIR / "conditions.csv"
PATIENTS_PATH = RAW_DATA_DIR / "patients.csv"
MEDICATIONS_PATH = RAW_DATA_DIR / "medications.csv"

T2DM_PATIENTS_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "t2dm_patients.csv"
METFORMIN_COHORT_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "metformin_cohort.csv"
NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "no_prior_hf_metformin_cohort.csv"
BASELINE_COVARIATES_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "baseline_covariates.csv"
COMPLETE_CASE_COHORT_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "complete_case_cohort.csv"

# ---------------------------------------------------------------------------
# T2DM inclusion logic: CODE-based (SNOMED-CT)
# EDA demonstrated that:
#   1. DESCRIPTION-based substring matching missed the base diagnosis due
#      to word-order ("Diabetes mellitus type 2" vs. "type 2 diabetes"
#      and a Roman-numeral variant ("type II").
#   2. Every CODE maps 1:1 to exactly one DESCRIPTION in this dataset, so
#      CODE-based matching is reliable and immune to phrasing variants.
#   3. 73.9% of patients with a T2DM-specific complication code never had
#      a base-diagnosis row, so requiring the base diagnosis alone would
#      wrongly exclude the majority of real T2DM patients.
# ---------------------------------------------------------------------------

T2DM_BASE_CODE = "44054006"  # Diabetes mellitus type 2 (disorder)

T2DM_COMPLICATION_CODES = {
    "1501000119109",    # Proliferative diabetic retinopathy due to type II diabetes mellitus
    "1551000119108",    # Nonproliferative diabetic retinopathy due to type 2 diabetes mellitus
    "157141000119108",  # Proteinuria due to type 2 diabetes mellitus
    "368581000119106",  # Neuropathy due to type 2 diabetes mellitus
    "422034002",        # Retinopathy due to type 2 diabetes mellitus
    "60951000119105",   # Blindness due to type 2 diabetes mellitus
    "90781000119102",   # Microalbuminuria due to type 2 diabetes mellitus
    "97331000119101",   # Macular edema and retinopathy due to type 2 diabetes mellitus
}

T2DM_INCLUSION_CODES = {T2DM_BASE_CODE} | T2DM_COMPLICATION_CODES

# ---------------------------------------------------------------------------
# Metformin monotherapy / new-user logic
# EDA demonstrated that:
#   1. Broad, unbiased discovery (all medications prescribed to T2DM
#      patients, ranked by frequency) and a targeted antidiabetic-name
#      pattern check across the full T2DM-cohort medication list agreed
#      exactly, confirming only 6 antidiabetic drugs exist in this
#      dataset at all - so the CODE list below is exhaustive, not a guess.
#   2. Metformin (CODE 860975) and an insulin 70/30 mix (CODE 106892) are
#      the only antidiabetic drugs with meaningful volume (3,995 and
#      10,527 patients respectively); the remaining 4 drugs are present
#      but rare (6-811 patients each).
#   3. Insulin glargine, detemir, and degludec are genuinely absent from
#      Synthea's diabetes module - confirmed by two independent search
#      methods, not a naming/matching miss.
#   4. Only 16 of 3,995 metformin-exposed T2DM patients (0.4%) had
#      another antidiabetic drug before metformin, so requiring metformin
#      to be each patient's first antidiabetic drug (new-user,
#      first-in-class-sequence definition) excludes very few patients
#      while correctly enforcing the new-user, active-comparator design.
# ---------------------------------------------------------------------------

METFORMIN_CODE = "860975"  # 24 HR Metformin hydrochloride 500 MG ER Oral Tablet
 
OTHER_ANTIDIABETIC_CODES = {
    "106892": "insulin isophane human 70 / insulin regular human 30 [Humulin]",
    "311034": "insulin regular human 100 UNT/ML Injectable Solution",
    "897122": "liraglutide 6 MG/ML Pen Injector",
    "865098": "Insulin Lispro 100 UNT/ML Injectable Solution [Humalog]",
    "1373463": "canagliflozin 100 MG Oral Tablet",
}
 
ANTIDIABETIC_CODES = {METFORMIN_CODE} | set(OTHER_ANTIDIABETIC_CODES.keys())

# ---------------------------------------------------------------------------
# No prior heart failure at baseline
# EDA demonstrated that:
#   1. Only 2 distinct heart failure related condition codes exist in this dataset,
#      and both map 1:1 with their DESCRIPTION.
#   2. "Chronic congestive heart failure" and "Heart failure" are treated
#      as equally disqualifying prior heart failure evidence for this exclusion -
#      likely a severity/staging distinction rather than two unrelated
#      conditions, but either one means the patient cannot experience
#      our incident heart failure hospitalization outcome as a new event.
# ---------------------------------------------------------------------------

HF_INCLUSION_CODES = {
    "88805009": "Chronic congestive heart failure (disorder)",
    "84114007": "Heart failure (disorder)",
}

# ---------------------------------------------------------------------------
# Baseline covariates
# EDA demonstrated that:
#   1. BMI, HbA1c, systolic BP, and diastolic BP each map 1:1 to a single
#      CODE/DESCRIPTION/UNITS combination - full coverage across all
#      3,964 cohort patients, no issues.
#   2. eGFR (CODE 33914-3) and creatinine (CODE 38483-4) each have
#      multiple DESCRIPTION/UNITS variants under the same CODE, and
#      their VALUE distributions genuinely differ between variants - 
#      the minority variants show implausible or systematically different values. 
#      Since the majority variant alone already covers 100% of the cohort, we use
#      only that variant for each and discard the rest, rather than
#      attempting to reconcile inconsistent data.
#   3. most patients' labs are recorded shortly after metformin start, not before - 
#      a timing artifact of Synthea's simulated encounter ordering, not missing data
#   4. eGFR and creatinine have long, implausible tails even within their single canonical variant.
#      Values outside the chosen plausible ranges are set to MISSING (NaN).
#   5. eGFR has 655 implausible  records tightly clustered at 1.0-1.9, and all 655
#      would fall within the plausible range if multiplied by 100.
#      Confirmed via LOINC/FHIR documentation that CODE 33914-3 has no
#      legitimate alternate scale/unit convention (defined as mL/min/1.73m2).
#      Values will be corrected rather than discarded.
#
# Baseline definition: each patient's most recent value not later than 40 days after their metformin_start_date
# ---------------------------------------------------------------------------

COVARIATE_OBSERVATION_CODES = {
    "39156-5": "bmi",
    "4548-4": "hba1c",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "33914-3": "egfr",
    "38483-4": "creatinine",
}

CANONICAL_VARIANT_FILTER = {
    "33914-3": (
        "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] "
        "in Serum or Plasma by Creatinine-based formula (MDRD)",
        "mL/min/{1.73_m2}",
    ),
    "38483-4": ("Creatinine [Mass/volume] in Blood", "mg/dL"),
}

BASELINE_WINDOW_DAYS_AFTER_INDEX = 40

PLAUSIBLE_RANGES = {
    "bmi": (10, 80),
    "hba1c": (3, 20),
    "systolic_bp": (60, 250),
    "diastolic_bp": (30, 150),
    "egfr": (2, 200),
    "creatinine": (0.1, 15),
}

SCALE_CORRECTION_CODES = {
    "33914-3": 100,  # egfr
}


def _strip_timezone(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """ Standardize dates to timezone-naive (UTC) for internal consistency """
    for col in date_columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df


def load_conditions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )
    return _strip_timezone(df, ["START", "STOP"])
 
 
def load_patients(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE"],
        parse_dates=["BIRTHDATE", "DEATHDATE"],
    )
    df = df.rename(columns={"Id": "patient_id"})
    return _strip_timezone(df, ["BIRTHDATE", "DEATHDATE"])
 
 
def load_medications(path: Path) -> pd.DataFrame:
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
    logging.info(f"Baseline window: DATE <= metformin_start_date + {BASELINE_WINDOW_DAYS_AFTER_INDEX} days")
    logging.info(f"Loading no-prior-heart-failure cohort from {NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH} ...")
    cohort = pd.read_csv(NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH, parse_dates=["metformin_start_date"])
    cohort = _strip_timezone(cohort, ["metformin_start_date"])
    patient_ids = set(cohort["patient_id"].unique())
    logging.info(f"  {len(patient_ids):,} cohort patients loaded")
 
    logging.info(f"Loading observations from {RAW_DATA_DIR / 'observations.csv'} ...")
    observations = load_baseline_observations(RAW_DATA_DIR / "observations.csv", patient_ids)
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
 
    complete_case.to_csv(COMPLETE_CASE_COHORT_OUTPUT_PATH, index=False)
    logging.info(f"Saved: {COMPLETE_CASE_COHORT_OUTPUT_PATH}")
