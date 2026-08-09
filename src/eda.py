""" Exploratory analysis supporting the T2DM cohort-identification logic in preprocess_data.py. """

import logging
import pandas as pd
from pathlib import Path

from .helpers import log_separator

RAW_DATA_DIR = Path("raw_data/csv")
CONDITIONS_PATH = RAW_DATA_DIR / "conditions.csv"

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
                 f"against all {len(description_to_codes)} distinct DESCRIPTIONs for diabetes-related conditions")
 
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
 

def run_eda():
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
    log_separator()