import sys
import logging
from pathlib import Path

from .config import (
    OUTPUT_DIR,
    RUN_DIR,
    LOG_FILE,
    PREPROCESSED_DATA_DIR,
    INJECTED_DATA_DIR,
    ANALYSIS_DIR,
    MONTE_CARLO_OUTPUT_DIR,
)


def setup_pipeline(monte_carlo=False):
    """ Create necessary directories and configure logging """
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info(f"Logging initialized. Run folder: {RUN_DIR}")
    logging.info(f"Writing log to: {LOG_FILE}")

    directories = (PREPROCESSED_DATA_DIR, INJECTED_DATA_DIR, ANALYSIS_DIR)
    if monte_carlo: directories = (MONTE_CARLO_OUTPUT_DIR, )
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    logging.info("Needed output directories created")


def log_separator(title=None):
    """ Logs a divider line, optionally with a centered title """
    logging.info("=" * 80)
    if title:
        logging.info(title.center(80))
        logging.info("=" * 80)


def find_latest_completed_run(required_file: str = "complete_case_cohort.csv") -> Path:
    """ Finds the most recent run folder under output/ that contains the given required artifact """
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"No {OUTPUT_DIR}/ directory found - run the main pipeline (scripts/run_pipeline.py) first")
 
    candidate_dirs = sorted(
        (d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d != RUN_DIR),
        reverse=True,
    )
 
    for candidate in candidate_dirs:
        candidate_file = candidate / "01_preprocessed_data" / required_file
        if candidate_file.exists():
            logging.info(f"Found most recent completed run: {candidate}")
            return candidate
 
    raise FileNotFoundError(f"No prior run folder under {OUTPUT_DIR}/ contains 01_preprocessed_data/{required_file} - run the main pipeline (scripts/run_pipeline.py) first")
