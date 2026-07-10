FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only (~200MB vs ~2GB for CUDA), sufficient for training on CPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# all source lives under cloud/ — self-contained
COPY model/     model/
COPY pipeline/  pipeline/
COPY serving/   serving/
COPY pull_and_build.py       .
COPY train_gcs.py            .
COPY export_item_vectors.py  .

# Generate gRPC stubs from proto at build time
RUN python -m grpc_tools.protoc -I/app/serving --python_out=/app/serving --grpc_python_out=/app/serving /app/serving/twotower_ranker.proto

ENV PYTHONUNBUFFERED=1

# Job 1: python /app/pull_and_build.py
# Job 2: python /app/train_gcs.py
# Job 3: python /app/export_item_vectors.py
# Serving: python /app/serving/server.py
