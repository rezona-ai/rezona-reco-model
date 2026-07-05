# Coarse-Ranking Model — Cloud Training

每日两步定时任务：
1. **Job 1（01:00 UTC）** BigQuery 拉取最近 N 天曝光日志 → 特征工程 → 上传 artifacts 到 GCS
2. **Job 2（02:00 UTC）** 从 GCS 拉 artifacts → 训练 TwoTowerLite → 上传 checkpoint 到 GCS

```
BigQuery (reco_labeled_log_flat)
  └─[pull_and_build.py]─→ GCS artifacts/latest/
                               └─[train_gcs.py]─→ GCS ckpts/YYYY-MM-DD/
```

---

## 目录结构

```
cloud/
├── pull_and_build.py       # Job 1: BQ 拉取 + 特征工程 + 上传 GCS
├── train_gcs.py            # Job 2: 下载特征 + 训练 + 上传 ckpt
├── Dockerfile              # 两个 Job 共用一个镜像
├── cronjob_features.yaml   # Job 1 CronJob（01:00 UTC）
├── cronjob.yaml            # Job 2 CronJob（02:00 UTC）
├── deploy.sh               # 一键 build + push + apply
└── requirements.txt        # Python 依赖
```

依赖的模型/pipeline 代码（容器内从同一镜像引用）：
- `model/two_tower_lite.py` — 模型定义
- `model/metrics.py` — auc / gauc
- `pipeline/build_features.py` — 特征工程（DATA_DIR 支持 env override）

---

## GCS 目录约定

```
gs://rezona-ml/
└── coarse-ranking/
    ├── artifacts/
    │   └── latest/          # build_features.py 产出，每次覆盖
    │       ├── train.npz
    │       ├── test.npz
    │       ├── config.json
    │       └── vocab.json
    └── ckpts/
        └── YYYY-MM-DD/      # 每次训练按日期写入
            ├── two_tower_lite_best.pt   # best-by-GAUC checkpoint
            ├── two_tower_lite.pt        # 最终 epoch checkpoint
            └── curves.csv               # 训练曲线数据
```

---

## 首次部署

### 1. 创建 GCS bucket

```bash
gsutil mb -l us-east1 gs://rezona-ml
```

### 2. 上传初始特征文件

```bash
gsutil -m cp artifacts_jul1/{train,test}.npz \
              artifacts_jul1/{config,vocab}.json \
  gs://rezona-ml/coarse-ranking/artifacts/latest/
```

### 3. 配置 IAM（Workload Identity）

```bash
PROJECT=YOUR_GCP_PROJECT_ID
SA_NAME=ml-trainer

# 创建 GCP Service Account
gcloud iam service-accounts create ${SA_NAME} \
  --display-name="ML Trainer"

# 赋予 GCS 读写权限
gsutil iam ch \
  serviceAccount:${SA_NAME}@${PROJECT}.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://rezona-ml

# 绑定 Workload Identity（K8s SA → GCP SA）
gcloud iam service-accounts add-iam-policy-binding \
  ${SA_NAME}@${PROJECT}.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT}.svc.id.goog[ml/ml-trainer]"

# 在集群里创建对应的 K8s ServiceAccount
kubectl create namespace ml
kubectl create serviceaccount ml-trainer -n ml
kubectl annotate serviceaccount ml-trainer -n ml \
  iam.gke.io/gcp-service-account=${SA_NAME}@${PROJECT}.iam.gserviceaccount.com
```

### 4. Build 镜像并部署 CronJob

```bash
./cloud/deploy.sh YOUR_GCP_PROJECT_ID
```

脚本会自动：build Docker 镜像 → push 到 GCR → 将镜像地址注入 cronjob.yaml → kubectl apply。

---

## 日常操作

### 手动触发一次训练

```bash
kubectl create job --from=cronjob/coarse-ranking-train manual-$(date +%m%d) -n ml
```

### 查看训练日志

```bash
kubectl logs -f job/manual-0705 -n ml
```

### 查看 CronJob 状态

```bash
kubectl get cronjob coarse-ranking-train -n ml
kubectl get jobs -n ml
```

### 查看已上传的 ckpt

```bash
gsutil ls gs://rezona-ml/coarse-ranking/ckpts/
```

### 下载最新 best checkpoint

```bash
DATE=$(gsutil ls gs://rezona-ml/coarse-ranking/ckpts/ | sort | tail -1 | tr -d '/')
gsutil cp ${DATE}/two_tower_lite_best.pt ./artifacts/ckpt/
```

---

## 环境变量

`train_gcs.py` 通过环境变量配置，在 `cronjob.yaml` 的 `env:` 块中设置：

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `GCS_BUCKET` | ✅ | — | GCS bucket 名称 |
| `GCS_ARTIFACTS` | ✅ | — | 特征文件 GCS 前缀 |
| `GCS_CKPT_PREFIX` | ✅ | — | ckpt 上传路径前缀 |
| `EPOCHS` | — | `2` | 训练轮数 |
| `BS` | — | `4096` | batch size |
| `LR` | — | `3e-3` | 学习率 |
| `EVAL_EVERY` | — | `50` | 每隔多少 step 评估一次 |

---

## 资源规格

| 配置 | CPU | Memory | 训练时长（估算）| 适用场景 |
|---|---|---|---|---|
| CPU（当前默认）| 4–8 核 | 32 Gi | ~30–60 分钟 | 成本低，日常重训 |
| GPU L4 | 4 核 | 16 Gi | ~5 分钟 | 快速迭代 |
| GPU T4 | 4 核 | 16 Gi | ~10 分钟 | 成本居中 |

内存瓶颈来自训练数据全量载入 torch（当前 13 天 / 411 万行 ≈ 7 GB）；
数据窗口扩大时线性增长，超过 30 天建议升至 64 Gi。

GPU 版本：取消 `cronjob.yaml` 中注释的 GPU 段，并切换到 GPU 镜像 tag。

---

## 特征文件更新

`build_features.py` 产出新特征后，覆盖写回 GCS `latest/` 即可，下次 CronJob 自动拉取：

```bash
TRAIN_DAYS=6/25,6/26,6/27,6/28,6/29,6/30,7/1 \
TEST_DAYS=7/2 \
OUT_DIR=artifacts_new \
python pipeline/build_features.py

gsutil -m cp artifacts_new/{train,test}.npz \
              artifacts_new/{config,vocab}.json \
  gs://rezona-ml/coarse-ranking/artifacts/latest/
```
