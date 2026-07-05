FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# all source lives under cloud/ — self-contained
COPY model/     model/
COPY pipeline/  pipeline/
COPY pull_and_build.py .
COPY train_gcs.py      .

ENV PYTHONUNBUFFERED=1

# Job 1: python /app/pull_and_build.py
# Job 2: python /app/train_gcs.py
