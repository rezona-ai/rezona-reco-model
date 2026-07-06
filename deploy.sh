#!/usr/bin/env bash
# Build 镜像（Cloud Build）、apply CronJob，并可选立即触发一次执行。
#
# 用法:
#   ./deploy.sh                  # build + push + apply（不立即执行）
#   ./deploy.sh --run-now        # build + push + apply + 立即触发两个 Job
set -euo pipefail

PROJECT="rezonaai"
IMAGE="us-east1-docker.pkg.dev/${PROJECT}/reco-model/ml-trainer:latest"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_NOW=false

for arg in "$@"; do
  [[ "$arg" == "--run-now" ]] && RUN_NOW=true
done

echo "==> [1/3] building image via Cloud Build: ${IMAGE}"
gcloud builds submit \
  --tag "${IMAGE}" \
  --machine-type=e2-highcpu-8 \
  "${REPO_DIR}"

echo "==> [2/3] (image pushed by Cloud Build)"

echo "==> [3/3] applying CronJobs"
kubectl apply -f "${REPO_DIR}/cronjob_features.yaml"
kubectl apply -f "${REPO_DIR}/cronjob.yaml"

if [[ "$RUN_NOW" == "true" ]]; then
  DATE=$(date +%m%d%H%M)
  echo "==> triggering immediate run"
  kubectl create job --from=cronjob/two-tower-lite-features \
    "features-now-${DATE}" -n production
  echo "    waiting for features job to complete..."
  kubectl wait --for=condition=complete \
    job/"features-now-${DATE}" -n production --timeout=3600s
  kubectl create job --from=cronjob/two-tower-lite-train \
    "train-now-${DATE}" -n production
  echo "    train job started: train-now-${DATE}"
  echo "    logs: kubectl logs -f job/train-now-${DATE} -n production"
fi

echo "==> done"
echo "    status:  kubectl get cronjob -n production"
echo "    manual:  kubectl create job --from=cronjob/two-tower-lite-features run-\$(date +%m%d) -n production"
