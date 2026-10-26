# reco-model 在 gtxy-rzn-infra 的部署清单

## 当前状态(2026-10-26 影子转正)

- **本目录是生产唯一清单来源**,`./deploy.sh` 只 apply 本目录到新集群 `gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1`。
  `two_tower_lite/*.yaml`、`swing/cronjob_swing.yaml` 与本目录内容保持一致,仅作历史保留,不要单独 apply。
- 新集群 `production` 跑:`two-tower-lite-features`(03:00 UTC)→ `two-tower-lite-train`(04:00)→ `two-tower-lite-export-items`(06:00)→ ranker 07:00 UTC 热加载;`swing-i2i-build`(04:30)。
- 旧集群 `gke_rezonaai_us-east1_autopilot-cluster-1` 的 4 个 CronJob(features/train/export-items/swing)已于 2026-10-26 **suspend**,ranker 早已 0/0。
- 事故背景:9-23 新集群 features 是从**未提交的本地 yaml** apply 的(写影子前缀 `_shadow_gtxy/`),train/export 未在新集群部署;旧集群 features 读旧项目 BQ(9-21 后冻结)每天失败,
  导致生产词表停在 9-22、游戏集合停在 9-21,9-22 后的新游戏 `predict_ctr` 全为 0,持续约 5 周。修复操作见 2026-10-26 TwoTower 修复 runbook。
- `gs://rezona-ml/_shadow_gtxy/` 已退役,`gs://rezona-ml/_backup_20261026/` 为修复前生产三件套备份;稳定运行 2 周后清理两者。
- **线上改动必须同步提交本目录**,否则下次 `deploy.sh` 会把线上改回去。

---

## 迁移记录(docs/28 阶段1,2026-09-23)

docs/28 阶段1。本目录是**改造 overlay**,原文件(`two_tower_lite/*`、`swing/*`)不动,便于对比与回滚(删本目录即回退)。

## 相对原 manifest 的改动(逐项)

| 改动 | 原 | 新 |
| --- | --- | --- |
| 镜像 | `us-east1-docker.pkg.dev/rezonaai/reco-model/ml-trainer` | `.../gtxy-rzn-infra/reco-model/ml-trainer` |
| BQ_PROJECT | `rezonaai` | `gtxy-rzn-infra`(读新项目 dwd/ads 每日复制副本,天级训练可接受 24h 滞后) |
| BQ_TABLE | `rezonaai.dwd/ads.*` | `gtxy-rzn-infra.dwd/ads.*` |
| swing REDIS_HOST | (未设 → 代码默认旧 `10.151.208.35`) | **显式 `10.97.68.75:6379`**(新 reco-redis,坑1) |
| KSA `recoleaf`/`syncer-sa` | 旧项目 WI | 新建 + 注解新 GSA(`00-serviceaccount.yaml`) |
| GCS_BUCKET | `rezona-ml`(不改,留旧项目,SA 已跨项目授权 ADR-014) | 同 |

## 部署前提(rezona-infra 侧已完成)
- AR `gtxy-rzn-infra/reco-model` 已建;GSA `recoleaf-kafka-writer`/`syncer` 已建并授权(BQ+Kafka+rezona-ml/datalake 桶+WI)。
- **镜像尚未推到新 AR** —— 见下第 1 步。

## 部署步骤(用户执行)

```bash
# 1. 构建镜像到新项目 AR(在 rezona-reco-model 仓库根)
gcloud builds submit --project=gtxy-rzn-infra \
  --tag us-east1-docker.pkg.dev/gtxy-rzn-infra/reco-model/ml-trainer:latest \
  --machine-type=e2-highcpu-8 .

# 2. KSA(先建,否则 pod 起不来)
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 apply -f deploy/gtxy/00-serviceaccount.yaml

# 3. 先验证 WI 链路(可选但建议):用 recoleaf KSA 跑一次性 pod 测 gcloud auth
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 -n production run wi-test \
  --image=google/cloud-sdk:slim --serviceaccount=recoleaf --restart=Never -it --rm -- gcloud auth list
# 期望 ACTIVE = recoleaf-kafka-writer@gtxy-rzn-infra.iam.gserviceaccount.com

# 4. 部署在线推理服务(ranker)—— 它只读 GCS,先起来
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 apply -f deploy/gtxy/serving_deployment.yaml
# ⚠️ 首次启动 ranker 会从 gs://rezona-ml 拉模型;若模型是旧项目训练产出的,直接可用(桶共享)

# 5. 手动触发一轮离线链路验证(不等 schedule),按依赖顺序:
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 apply -f deploy/gtxy/cronjob_features.yaml
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 apply -f deploy/gtxy/cronjob.yaml
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 apply -f deploy/gtxy/cronjob_export_items.yaml
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 apply -f deploy/gtxy/cronjob_swing.yaml
# 手动跑一次 features(create job from cronjob):
kubectl --context=gke_gtxy-rzn-infra_us-east1_autopilot-cluster-1 -n production create job --from=cronjob/two-tower-lite-features feat-manual-1
```

## 影子验证判据(阶段1 通过标准)
- `wi-test` 输出正确 GSA;
- features/train/export-items 手动跑通(exit 0),产物写入 `gs://rezona-ml/two_tower_lite/{artifacts,ckpts,item_vectors}`(可与旧项目产出对比大小/时间);
- swing 跑通,写入新 reco-redis `swing:i2i:*`(`redis-cli -h 10.97.68.75 --scan --pattern 'swing:i2i:*' | head`);
- ranker pod Ready(从 GCS 加载模型成功),gRPC :50051 可连。

## ⚠️ 注意
- **阶段1 不接线上流量**:ranker 部署在新集群但 reco-leaf(阶段2)还没迁,没有调用方,纯验证产物正确性。
- 与旧项目**并行运行**:旧集群的 cronjob/ranker 继续跑,互不干扰(读同一 GCS/BQ,写 GCS 同前缀会覆盖 —— 若担心污染旧产物,可让新链路写独立 GCS 前缀验证,阶段4 切换时再统一;当前建议先只手动跑一次对比、不开新 schedule,避免与旧 schedule 双写)。
- **swing 双写 feature_redis 风险**:新 swing 写新 reco-redis(10.97.68.75),旧 swing 写旧(10.151.208.35),互不干扰 ✓。
