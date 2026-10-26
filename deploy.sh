#!/usr/bin/env bash
# Build shared image (Cloud Build) and apply all CronJobs + serving.
#
# 2026-10-26 起生产在 gtxy-rzn-infra（新集群），唯一清单来源是 deploy/gtxy/。
# 旧项目 rezonaai 的 CronJob 已暂停，切勿再往旧集群 apply。
#
# Usage:
#   ./deploy.sh                  # build + apply all
#   ./deploy.sh --run-now        # + immediately trigger two_tower_lite pipeline
set -euo pipefail

PROJECT="gtxy-rzn-infra"
CONTEXT="gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1"
NS="production"
IMAGE="us-east1-docker.pkg.dev/${PROJECT}/reco-model/ml-trainer:latest"
ROOT="$(cd "$(dirname "$0")" && pwd)"
MANIFESTS="${ROOT}/deploy/gtxy"
RUN_NOW=false

for arg in "$@"; do
  [[ "$arg" == "--run-now" ]] && RUN_NOW=true
done

kc() { kubectl --context "${CONTEXT}" "$@"; }

echo "==> [1/3] building image via Cloud Build: ${IMAGE}"
gcloud builds submit \
  --project "${PROJECT}" \
  --tag "${IMAGE}" \
  --machine-type=e2-highcpu-8 \
  "${ROOT}"

echo "==> [2/3] (image pushed by Cloud Build)"

echo "==> [3/3] applying CronJobs + serving to ${CONTEXT}"
kc apply -f "${MANIFESTS}/00-serviceaccount.yaml"
kc apply -f "${MANIFESTS}/cronjob_features.yaml"
kc apply -f "${MANIFESTS}/cronjob.yaml"
kc apply -f "${MANIFESTS}/cronjob_export_items.yaml"
kc apply -f "${MANIFESTS}/cronjob_swing.yaml"
kc apply -f "${MANIFESTS}/serving_deployment.yaml"

if [[ "${RUN_NOW}" == "true" ]]; then
  DATE=$(date +%m%d%H%M)
  echo "==> triggering immediate two_tower_lite run"
  kc create job --from=cronjob/two-tower-lite-features \
    "features-now-${DATE}" -n "${NS}"
  echo "    waiting for features job to complete..."
  kc wait --for=condition=complete \
    job/"features-now-${DATE}" -n "${NS}" --timeout=3600s
  kc create job --from=cronjob/two-tower-lite-train \
    "train-now-${DATE}" -n "${NS}"
  echo "    train job started: train-now-${DATE}"
  echo "    ⚠️ train 完成前不要重启 ranker（新词表 + 旧 ckpt 会 size mismatch）"
  echo "    logs: kubectl --context ${CONTEXT} logs -f job/train-now-${DATE} -n ${NS}"
fi

echo "==> done"
echo "    status:  kubectl --context ${CONTEXT} get cronjob -n ${NS}"
