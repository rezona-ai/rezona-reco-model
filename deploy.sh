#!/usr/bin/env bash
# Build shared image (Cloud Build) and apply all CronJobs + serving.
#
# Usage:
#   ./deploy.sh                  # build + apply all
#   ./deploy.sh --run-now        # + immediately trigger two_tower_lite pipeline
set -euo pipefail

PROJECT="rezonaai"
IMAGE="us-east1-docker.pkg.dev/${PROJECT}/reco-model/ml-trainer:latest"
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_NOW=false

for arg in "$@"; do
  [[ "$arg" == "--run-now" ]] && RUN_NOW=true
done

echo "==> [1/3] building image via Cloud Build: ${IMAGE}"
gcloud builds submit \
  --tag "${IMAGE}" \
  --machine-type=e2-highcpu-8 \
  "${ROOT}"

echo "==> [2/3] (image pushed by Cloud Build)"

echo "==> [3/3] applying CronJobs + serving"
kubectl apply -f "${ROOT}/two_tower_lite/cronjob_features.yaml"
kubectl apply -f "${ROOT}/two_tower_lite/cronjob.yaml"
kubectl apply -f "${ROOT}/two_tower_lite/cronjob_export_items.yaml"
kubectl apply -f "${ROOT}/swing/cronjob_swing.yaml"
kubectl apply -f "${ROOT}/two_tower_lite/serving/deployment.yaml"

if [[ "${RUN_NOW}" == "true" ]]; then
  DATE=$(date +%m%d%H%M)
  echo "==> triggering immediate two_tower_lite run"
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
