<p style="font-size: 2em; font-weight: 700; margin-bottom: 0;">
  Real-World Evidence Against Ground Truth
</p>
<p style="font-size: 1.5em; font-weight: 600; margin-top: 0;">
  Validating a Causal Inference Pipeline on Semi-Synthetic EHR Data
</p>

#

A real-world evidence pipeline that runs a target trial emulation estimating the comparative effect of SGLT2 inhibitors vs. DPP-4 inhibitors on heart failure hospitalization risk in type 2 diabetes patients. Because the true effect is investigator-injected on top of real Synthea baseline covariates, the pipeline is used primarily to prove that a propensity-score causal pipeline (stabilized IPTW, doubly-robust estimation) correctly recovers a known treatment effect from confounded observational data, including under a misspecified propensity model.

The complete methodology, including the injection mechanism, the causal estimators, the 500-seed Monte Carlo validation, and an explicit treatment of the project's assumptions and limitations, is presented in `reports\Technical_report.pdf` while a compact summary of the study is given in `reports\1_page_report.pdf`.

#### Keywords:
Real-world evidence; Target trial emulation; Propensity score methods; Inverse probability of treatment weighting; Doubly-robust estimation; Synthetic EHR data

---

## Pipeline Architecture

```
                    Synthea (synthetic EHR)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 · eda.py                Exploratory Analysis       │
│  Code vs. free-text consistency · diagnosis/complication    │
│  overlap · medication frequency audit · lab-to-index        │
│  distance percentiles · covariate correlation matrix        |
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 · preprocess_data.py    Cohort Construction        │
│  SNOMED-coded T2DM · metformin new-user · no prior HF       │
│  eGFR ×100 unit-scaling fix · same-timestamp duplicate lab  │
│  values averaged · implausible-value exclusion              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 · injection.py          Ground-Truth Injection     │
│  Treatment: logistic model on standardized covariates,      │
│  calibrated to ~45% SGLT2i uptake                           │
│  Outcome: exponential PH simulation, true HR = 0.67,        │
│  calibrated to ~10% 2-year cumulative incidence             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 · analysis.py           Causal Estimation          │
│  Naive · stabilized IPTW · doubly-robust (IPTW + Cox)       │
│  SMD covariate balance + Love plot · negative control       │
│  E-value · propensity misspecification sensitivity analysis │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 · monte_carlo.py        Calibration Validation     │
│  500-seed replication of injection + analysis               │
│  Empirical CI coverage per estimator vs. true HR = 0.67     │
│  Pathological-seed detection                                │
│  + quasi-separation / weight-instability root-cause diag    │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Source

| Source | Access | Description |
|---|---|---|
| Synthea v3.3.0 | Local synthetic generator (Docker) | ~115k-patient synthetic EHR/claims cohort, Illinois |

No real patient data is used anywhere in this project.

## Setup

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Clone the repository
git clone https://github.com/vg-portfolio-26/real_world_evidence.git
cd rwe

# 2. Generate the synthetic Synthea cohort
docker compose run synthea

# 3. Build the pipeline image
docker compose build pipeline

# 4. Run the main pipeline (cohort build, injection, analysis)
docker compose run pipeline scripts/run_pipeline.py

# 5. Run the 500-seed Monte Carlo validation (reuses the latest completed cohort)
docker compose run pipeline scripts/run_monte_carlo.py
```

Python dependencies (managed inside the container):
Python · pandas 2.3.3 · numpy 2.4.6 · lifelines 0.30.3 · statsmodels 0.14.6 · matplotlib 3.11.0

---

## Project Structure

```
raw_data/
└── ...                           # Synthea-generated

scripts/
├── run_pipeline.py               # Main pipeline entrypoint
└── run_monte_carlo.py            # Monte Carlo validation entrypoint

src/
├── pipeline.py                   # Orchestration
├── config.py                     # Cohort filters, covariates, calibration targets, paths
├── eda.py                        # Stage 1: exploratory data checks
├── preprocess_data.py            # Stage 2: cohort construction, data-quality fixes
├── injection.py                  # Stage 3: ground-truth treatment/outcome injection
├── analysis.py                   # Stage 4: naive/IPTW/doubly-robust estimation, E-value
├── monte_carlo.py                # Stage 5: 500-seed calibration validation
└── helpers.py                    # Logging and shared utilities

docker/
├── Dockerfile                    # Pipeline image
├── Dockerfile.synthea            # Synthea data-generation image
└── synthea.config                # Synthea generator configuration
```

---

## License

Repo and content not licensed for use or redistribution.
