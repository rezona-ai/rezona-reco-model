#!/usr/bin/env bash
# 将本地历史 ndjson 上传到 GCS raw/ 作为缓存，避免 Job 1 重复从 BQ 拉取。
# 只需首次运行一次。
#
# 用法: ./cloud/upload_data.sh [data_dir]
set -euo pipefail

DATA_DIR=${1:-"$(cd "$(dirname "$0")/.." && pwd)/data"}
BUCKET="gs://rezona-ml/coarse-ranking/raw"

echo "==> uploading ndjson from ${DATA_DIR}/ to ${BUCKET}/"
gsutil -m cp "${DATA_DIR}"/*.ndjson "${BUCKET}/"
echo "==> done:"
gsutil ls "${BUCKET}/"
