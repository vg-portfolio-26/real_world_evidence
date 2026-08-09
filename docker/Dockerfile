# ---------------------------------------------------------------------------
# Main analysis pipeline image
#
# Raw Synthea data must exist in raw_data/ (generated via Dockerfile.synthea)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

VOLUME /app/raw_data
VOLUME /app/output

ENTRYPOINT ["python"]
CMD ["scripts/run_pipeline.py"]