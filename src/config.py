from datetime import datetime
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Raw input data (Synthea CSV export)
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path("raw_data/csv")

CONDITIONS_PATH = RAW_DATA_DIR / "conditions.csv"
PATIENTS_PATH = RAW_DATA_DIR / "patients.csv"
MEDICATIONS_PATH = RAW_DATA_DIR / "medications.csv"
OBSERVATIONS_PATH = RAW_DATA_DIR / "observations.csv"

# ---------------------------------------------------------------------------
# Run/output directory structure - one timestamped folder per pipeline run
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / RUN_TIMESTAMP

LOG_FILE = RUN_DIR / "pipeline.log"

SRC_SNAPSHOT_DIR = RUN_DIR / "00_src"

PREPROCESSED_DATA_DIR = RUN_DIR / "01_preprocessed_data"
INJECTED_DATA_DIR = RUN_DIR / "02_injected_data"
ANALYSIS_DIR = RUN_DIR / "03_analysis"
MONTE_CARLO_OUTPUT_DIR = RUN_DIR / "04_monte_carlo_results"

# ---------------------------------------------------------------------------
# Preprocessing outputs
# ---------------------------------------------------------------------------

T2DM_PATIENTS_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "t2dm_patients.csv"
METFORMIN_COHORT_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "metformin_cohort.csv"
NO_PRIOR_HF_METFORMIN_COHORT_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "no_prior_hf_metformin_cohort.csv"
BASELINE_COVARIATES_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "baseline_covariates.csv"

COMPLETE_CASE_COHORT_PATH = PREPROCESSED_DATA_DIR / "complete_case_cohort.csv"

# ---------------------------------------------------------------------------
# Injection outputs
# ---------------------------------------------------------------------------

TREATMENT_ASSIGNMENT_OUTPUT_PATH = INJECTED_DATA_DIR / "treatment_assignment.csv"
HF_OUTCOME_OUTPUT_PATH = INJECTED_DATA_DIR / "hf_outcome.csv"

# ---------------------------------------------------------------------------
# Analysis outputs
# ---------------------------------------------------------------------------

PROPENSITY_SCORES_OUTPUT_PATH = ANALYSIS_DIR / "propensity_scores.csv"
RESULTS_SUMMARY_OUTPUT_PATH = ANALYSIS_DIR / "results_summary.csv"
LOVE_PLOT_OUTPUT_PATH = ANALYSIS_DIR / "love_plot.png"
KM_CURVES_OUTPUT_PATH = ANALYSIS_DIR / "km_curves.png"
NEGATIVE_CONTROL_OUTPUT_PATH = ANALYSIS_DIR / "negative_control_results.csv"

# ---------------------------------------------------------------------------
# Monte Carlo outputs
# ---------------------------------------------------------------------------

MONTE_CARLO_RESULTS_PATH = MONTE_CARLO_OUTPUT_DIR / "monte_carlo_results.csv"
MONTE_CARLO_PLOT_PATH = MONTE_CARLO_OUTPUT_DIR / "monte_carlo_hr_distribution.png"
MONTE_CARLO_LOVE_PLOT_PATH = MONTE_CARLO_OUTPUT_DIR / "monte_carlo_love_plot.png"

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

# ---------------------------------------------------------------------------
# EDA-only exploratory patterns and reference lists (src/eda.py)
# These support the discovery/justification steps behind the inclusion logic
# above; they are not used for cohort filtering themselves.
# ---------------------------------------------------------------------------

EXPLORATORY_DIABETES_PATTERN = "diabetes"
BASE_DX_DESCRIPTION = "Diabetes mellitus type 2 (disorder)"
COMPLICATION_PATTERN = "type 2 diabetes|type II diabetes"
EXPLORATORY_HF_PATTERN = "heart failure"

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

ANTIDIABETIC_PATTERN = (
    "metformin|insulin|glipizide|glyburide|glimepiride|"
    "gliflozin|gliptin|glitazone|liraglutide|semaglutide|"
    "exenatide|dulaglutide|glargine|detemir|degludec"
)

BASELINE_COVARIATE_PATTERN = (
    "body mass index|bmi|glomerular filtration|egfr|creatinine|"
    "blood pressure|systolic|diastolic|hemoglobin a1c|hba1c"
)

# ---------------------------------------------------------------------------
# Injection: treatment assignment (src/injection.py)
#
# Logistic model coefficients for P(SGLT2i), on STANDARDIZED covariates (z-scores relative to this cohort's own mean/std).
# Positive coefficient = higher value of that covariate increases probability of SGLT2i (vs. DPP-4i).
#
#   age:           negative        - older patients more likely DPP-4i
#   bmi:           positive        - higher BMI more likely SGLT2i (weight-loss indication)
#   egfr:          positive        - better renal function more likely SGLT2i
#   hba1c:         ~0              - HbA1c doesn't strongly differentiate the choice between these two classes
#   systolic_bp:   slight positive - modest cardiometabolic-risk signal
#   creatinine:    negative        - worse renal function (higher creatinine) less likely SGLT2i, consistent with egfr
#
# Intercept is set to target an overall ~40-50% SGLT2i assignment rate
# at the cohort's mean covariate values (calibrated empirically once the model is run).
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

ASSIGNMENT_COEFFICIENTS = {
    "age": -0.35,
    "bmi": 0.30,
    "egfr": 0.40,
    "hba1c": 0.05,
    "systolic_bp": 0.10,
    "creatinine": -0.25,
}

COVARIATE_COLUMNS = list(ASSIGNMENT_COEFFICIENTS.keys())

# ---------------------------------------------------------------------------
# Injection: outcome (src/injection.py)
#
# Injected outcome: incident heart failure hospitalization.
# Model: exponential proportional-hazards simulation.
#
#   hazard_i = lambda_0 * exp(beta_treat * I(SGLT2i) + sum(beta_k * covariate_k_std))
#
# TRUE treatment effect (beta_treat) is set to reflect the real,
# published SGLT2i cardioprotective effect on HF hospitalization
# (EMPA-REG OUTCOME, DECLARE-TIMI 58, CVD-REAL: HR approx 0.65-0.70).
# This is the ground-truth effect the causal pipeline should recover later.
#
# Covariate coefficients: age, HbA1c, systolic BP, and creatinine increase heart failure risk;
# eGFR is protective (higher = lower risk); BMI has a modest positive association.
# ---------------------------------------------------------------------------

TRUE_SGLT2I_HAZARD_RATIO = 0.67  # midpoint of EMPA-REG/DECLARE/CVD-REAL range
TRUE_SGLT2I_LOG_HR = np.log(TRUE_SGLT2I_HAZARD_RATIO)

OUTCOME_COEFFICIENTS = {
    "age": 0.30,
    "bmi": 0.15,
    "egfr": -0.35,
    "hba1c": 0.20,
    "systolic_bp": 0.15,
    "creatinine": 0.25,
}

FOLLOWUP_DAYS = 365 * 2   # 2-year administrative censoring horizon
TARGET_OVERALL_EVENT_RATE = 0.10  # ~10% cumulative heart failure hospitalization incidence over 2 years

# ---------------------------------------------------------------------------
# Analysis (src/analysis.py)
# ---------------------------------------------------------------------------

NEGATIVE_CONTROL_TRUE_LOG_HR = 0.0  # falsification test: no true treatment effect on this simulated outcome

# ---------------------------------------------------------------------------
# Monte Carlo validation (src/monte_carlo.py)
# ---------------------------------------------------------------------------

N_SEEDS = 50
BASE_SEED = 1000

PATHOLOGICAL_HR_BOUNDS = (0.05, 5.0)  # flags a seed's doubly-robust HR as implausible outside this range
