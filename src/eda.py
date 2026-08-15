""" Exploratory analysis supporting the T2DM cohort-identification logic in preprocess_data.py. """

import logging
import pandas as pd
from pathlib import Path

from .helpers import log_separator

RAW_DATA_DIR = Path("raw_data/csv")
CONDITIONS_PATH = RAW_DATA_DIR / "conditions.csv"
MEDICATIONS_PATH = RAW_DATA_DIR / "medications.csv"

EXPLORATORY_DIABETES_PATTERN = "diabetes"
BASE_DX_DESCRIPTION = "Diabetes mellitus type 2 (disorder)"
COMPLICATION_PATTERN = "type 2 diabetes|type II diabetes"

# Explicitly excluded:
#   - 15777000  (Prediabetes): distinct, earlier-stage condition
#   - 127013003 (Disorder of kidney due to diabetes mellitus): does not specify type 1 vs type 2
#   - 427089005 (Diabetes from Cystic Fibrosis): not T2DM
#   - Z13.1 (Encounter for screening for diabetes mellitus): not a diagnosis
T2DM_EXCLUDED_CODES = {
    "15777000": "Prediabetes",
    "127013003": "Disorder of kidney due to diabetes mellitus (disorder) - type-ambiguous",
    "427089005": "Diabetes from Cystic Fibrosis - distinct disease entity",
    "Z13.1": "Encounter for screening for diabetes mellitus - not diagnosis",
}


def load_conditions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )
    return df


def explore_diabetes_related_descriptions(conditions: pd.DataFrame) -> None:
    """
    Log every unique DESCRIPTION containing 'diabetes' (case-insensitive),
    with counts, sorted descending. This is the broad grep that revealed
    the base-diagnosis wording mismatch and the ambiguous/ excluded categories.
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
    to determine whether requiring the base diagnosis alone would wrongly exclude real T2DM patients.
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
    logging.info("Excluded / ambiguous codes (not counted as T2DM evidence alone):")
    for code, reason in T2DM_EXCLUDED_CODES.items():
        subset = conditions.loc[conditions["CODE"].astype(str) == code]
        n_patients = subset["PATIENT"].nunique()
        logging.info(
            f"  CODE {code} ({reason}): {len(subset):,} records, {n_patients:,} unique patients"
        )


def check_code_vs_description_matching(conditions: pd.DataFrame) -> None:
    """ Investigate whether SYSTEM.CODE is a more reliable identifier than free-text DESCRIPTION matching for T2DM-related conditions. """
    logging.info("Checking CODE vs DESCRIPTION matching for diabetes-related conditions ...")
    mask = conditions["DESCRIPTION"].str.contains(
        EXPLORATORY_DIABETES_PATTERN, case=False, na=False
    )
    diabetes_related = conditions.loc[mask, ["CODE", "DESCRIPTION"]]
 
    code_to_descriptions = diabetes_related.groupby("CODE")["DESCRIPTION"].unique()
    description_to_codes = diabetes_related.groupby("DESCRIPTION")["CODE"].unique()
 
    logging.info(f"Checked all {len(code_to_descriptions)} distinct CODEs "
                 f"against all {len(description_to_codes)} distinct DESCRIPTIONs")
 
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


def load_medications(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["START", "STOP", "PATIENT", "DESCRIPTION", "CODE"],
        parse_dates=["START", "STOP"],
    )
    return df


def run_eda_conditions():
    logging.info(f"Loading conditions from {CONDITIONS_PATH} ...")
    conditions = load_conditions(CONDITIONS_PATH)
    logging.info(f"  {len(conditions):,} total condition records loaded")
    log_separator()

    explore_diabetes_related_descriptions(conditions)
    log_separator()

    compare_base_dx_vs_complication_only(conditions)
    log_separator()

    log_excluded_categories(conditions)
    log_separator()

    check_code_vs_description_matching(conditions)


def explore_medications_for_t2dm_cohort(t2dm_patient_ids: set) -> None:
    """ Full audit of every medication prescribed to T2DM-cohort patients """
    logging.info(f"Loading medications from  {MEDICATIONS_PATH} ...")
    medications = load_medications(MEDICATIONS_PATH)
    logging.info(f"  {len(medications):,} total medication records loaded")
    log_separator()
 
    cohort_meds = medications.loc[medications["PATIENT"].isin(t2dm_patient_ids)]
    logging.info(f"  {len(cohort_meds):,} medication records belong to T2DM-cohort patients")
 
    summary = (
        cohort_meds.groupby(["CODE", "DESCRIPTION"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
 
    logging.info(f"Top 10 medications by unique-patient count within the T2DM cohort (all {len(summary)} distinct drugs):")
    for _, row in summary.head(10).iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records"
        )

    return cohort_meds


def check_for_missed_antidiabetic_drugs(cohort_meds: pd.DataFrame) -> None:
    """ Targeted check across ALL medications for T2DM-cohort patients, specifically searching for antidiabetic drug name patterns. """
    antidiabetic_pattern = (
        "metformin|insulin|glipizide|glyburide|glimepiride|"
        "gliflozin|gliptin|glitazone|liraglutide|semaglutide|"
        "exenatide|dulaglutide|glargine|detemir|degludec"
    )
    mask = cohort_meds["DESCRIPTION"].str.contains(antidiabetic_pattern, case=False, na=False)
    matched = cohort_meds.loc[mask]
 
    summary = (
        matched.groupby(["CODE", "DESCRIPTION"])
        .agg(n_records=("PATIENT", "size"), n_patients=("PATIENT", "nunique"))
        .reset_index()
        .sort_values("n_patients", ascending=False)
    )
 
    logging.info(f"Antidiabetic-pattern check across ALL {len(cohort_meds):,} T2DM-cohort medication records:")
    logging.info(f"  {len(summary)} distinct drugs matched:")
    for _, row in summary.iterrows():
        logging.info(
            f"  CODE {row['CODE']}: {row['DESCRIPTION']} "
            f"- {row['n_patients']:,} patients, {row['n_records']:,} records"
        )

 
def run_eda_medications():
    from .preprocess_data import PREPROCESSED_DATA_DIR
 
    t2dm_path = PREPROCESSED_DATA_DIR / "t2dm_patients.csv"
    t2dm_patients = pd.read_csv(t2dm_path, usecols=["patient_id"])
    t2dm_patient_ids = set(t2dm_patients["patient_id"].unique())
    logging.info(f"Loaded {len(t2dm_patient_ids):,} T2DM patient IDs from {t2dm_path}")
 
    cohort_meds = explore_medications_for_t2dm_cohort(t2dm_patient_ids)
    log_separator()
    check_for_missed_antidiabetic_drugs(cohort_meds)