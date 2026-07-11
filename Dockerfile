FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only (~200MB vs ~2GB for CUDA), sufficient for serving and training on CPU nodes
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── two_tower_lite ────────────────────────────────────────────────────────────
COPY two_tower_lite/                      two_tower_lite/

# ── swing ─────────────────────────────────────────────────────────────────────
COPY swing/                               swing/

# Generate gRPC stubs from proto at build time
RUN python -m grpc_tools.protoc \
    -I/app/two_tower_lite/serving \
    --python_out=/app/two_tower_lite/serving \
    --grpc_python_out=/app/two_tower_lite/serving \
    /app/two_tower_lite/serving/twotower_ranker.proto

ENV PYTHONUNBUFFERED=1

# Entry points:
#   python /app/two_tower_lite/pull_and_build.py        features
#   python /app/two_tower_lite/train_gcs.py             training
#   python /app/two_tower_lite/export_item_vectors.py   item vector export
#   python /app/two_tower_lite/serving/server.py        gRPC serving
#   python /app/swing/run.py                            swing i2i build
