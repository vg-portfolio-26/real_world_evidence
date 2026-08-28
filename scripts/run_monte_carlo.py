import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.helpers import setup_pipeline, log_separator
from src.monte_carlo import run_monte_carlo_validation

if __name__ == "__main__":
    setup_pipeline(monte_carlo=True)
    log_separator("Starting Monte Carlo validation")
    run_monte_carlo_validation()
    log_separator("Monte Carlo validation finished")
