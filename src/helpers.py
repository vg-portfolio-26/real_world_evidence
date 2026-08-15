import sys
import logging
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("output")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / RUN_TIMESTAMP

LOG_FILE = RUN_DIR / "pipeline.log"

PREPROCESSED_DATA_DIR = RUN_DIR / "01_preprocessed_data"


def setup_pipeline():
    """Create necessary directories and configure logging."""
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

    for d in (PREPROCESSED_DATA_DIR,):
        d.mkdir(parents=True, exist_ok=True)
        logging.info("Needed output directories created")


def log_separator(title=None):
    logging.info("=" * 80)
    if title:
        logging.info(title)
        logging.info("=" * 80)