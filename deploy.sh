#!/usr/bin/env bash
# Build 镜像、push 到 GCR、apply CronJob，并可选立即触发一次执行。
#
# 用法:
#   ./cloud/deploy.sh                  # build + push + apply（不立即执行）
#   ./cloud/deploy.sh --run-now        # build + push + apply + 立即触发两个 Job
set -euo pipefail

PROJECT="rezonaai"
IMAGE="gcr.io/${PROJECT}/coarse-ranking-trainer:latest"
CLOUD_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_NOW=false

for arg in "$@"; do
  [[ "$arg" == "--run-now" ]] && RUN_NOW=true
done

echo "==> [1/3] building image: ${IMAGE}"
docker build -f "${CLOUD_DIR}/Dockerfile" \
             -t "${IMAGE}" \
             "${CLOUD_DIR}"

echo "==> [2/3] pushing image"
docker push "${IMAGE}"

echo "==> [3/3] applying CronJobs"
kubectl apply -f "${CLOUD_DIR}/cronjob_features.yaml"
kubectl apply -f "${CLOUD_DIR}/cronjob.yaml"

if [[ "$RUN_NOW" == "true" ]]; then
  DATE=$(date +%m%d%H%M)
  echo "==> triggering immediate run"
  kubectl create job --from=cronjob/coarse-ranking-features \
    "features-now-${DATE}" -n production
  echo "    waiting for features job to complete..."
  kubectl wait --for=condition=complete \
    job/"features-now-${DATE}" -n production --timeout=3600s
  kubectl create job --from=cronjob/coarse-ranking-train \
    "train-now-${DATE}" -n production
  echo "    train job started: train-now-${DATE}"
  echo "    logs: kubectl logs -f job/train-now-${DATE} -n production"
fi

echo "==> done"
echo "    status:  kubectl get cronjob -n production"
echo "    manual:  kubectl create job --from=cronjob/coarse-ranking-features run-\$(date +%m%d) -n production"
