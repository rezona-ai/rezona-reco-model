# TwoTowerLite 粗排推理服务

## 定位

粗排阶段：从召回集（数百到数千个候选游戏）中实时打分排序，供后续精排或截断使用。

```
召回服务 ──候选 game_ids──▶  [本服务]  ──(game_id, score)──▶  精排 / 截断
         user context ──────────────▶
```

---

## 目录结构

```
serving/
├── twotower_ranker.proto     # gRPC 接口定义
├── gen_proto.sh              # 本地生成 pb2 文件（开发用）
├── server.py                 # gRPC Server 主体，负责启动、模型加载、每日刷新
├── encoder.py                # user 侧特征编码（与训练 build_features.py 对齐）
├── item_store.py             # item vector 内存管理、打分、日志
├── deployment.yaml           # GKE Deployment + ClusterIP Service
└── README.md                 # 本文档
```

proto 生成的 `twotower_ranker_pb2.py` / `twotower_ranker_pb2_grpc.py` 在 Docker 构建时自动生成（`Dockerfile` 中 `RUN grpc_tools.protoc ...`），不提交到 git。

---

## Proto 接口

```protobuf
syntax = "proto3";
package twotower_ranker;

message TwoTowerUserFeatures {
  string user_id      = 1;   // 用于日志追踪，不参与模型计算
  string country_code = 2;   // e.g. "CN" / "US"
  string platform     = 3;   // "ios" / "android"
  string app_version  = 4;   // e.g. "1.87.0+2026070117"
}

message TwoTowerScoreRequest {
  TwoTowerUserFeatures user     = 1;
  repeated int64       game_ids = 2;   // 候选集，顺序不影响结果
}

message TwoTowerGameScore {
  int64 game_id = 1;
  float score   = 2;   // sigmoid 后的预估播放概率 [0, 1]
}

message TwoTowerScoreResponse {
  repeated TwoTowerGameScore scores     = 1;   // 与请求 game_ids 顺序对应
  string                     model_date = 2;   // item vectors 来自哪天，方便 debug
}

service TwoTowerRankerService {
  rpc Score(TwoTowerScoreRequest) returns (TwoTowerScoreResponse);
}
```

**设计说明**

- `user_id` 仅写入日志，不进入模型（TwoTowerLite user tower 只用 country/platform/app_version 三个 context 特征）。
- `game_id` 使用 `int64`，与 BigQuery 源表类型一致，比 string 节省约 30% 序列化体积。
- 返回值是 sigmoid 概率，调用方可直接用作排序分，无需再做变换。
- `model_date` 标识当次响应使用的 item vector 版本，便于排查线上与离线评估对不齐的问题。
- 响应顺序与请求 `game_ids` 严格对应，调用方无需重排。
- 不在 item vector 库中的 `game_id`（新游戏或已下架）返回 `score=0.0`，由调用方决定是否过滤。

> **向后兼容注意**：`game_id` 字段编号已固定（field 1），调用方升级 proto 时需同步部署，不可滚动灰度。

---

## 请求处理流程

```
TwoTowerScoreRequest
        │
        ▼
  encoder.py  UserEncoder.encode(country_code, platform, app_version)
        │  ① 查 vocab：token → int_id，未见 token → OOV (id=1)
        │  ② model.user_vec(batch)  ← 小 MLP，~0.1 ms
        ▼
  user_vec: float32 (1, 64)
        │
        ▼
  item_store.py  ServingBundle.score(game_ids)
        │  ① game_id(int64) → idx  (dict, O(1)/id)
        │  ② item_vecs[idxs]  numpy fancy indexing（GIL 释放，可并行）
        │  ③ 不在词表的 game_id → score=0.0
        ▼
  logits = item_vecs @ user_vec.T          # (K, 64) × (64, 1) = (K,)
  scores = sigmoid(logits)                 # float32 (K,)
        │
        ▼
  TwoTowerScoreResponse(scores=[...], model_date=...)
```

整个请求路径无网络 I/O，item vector 和模型权重全部常驻内存。

---

## 特征编码（encoder.py）

`UserEncoder` 在服务启动时从 `vocab.json` 中读取 user 侧三个特征的词表：

| 特征 | vocab key | 示例 token |
|---|---|---|
| `country_code` | `country` | `"CN"`, `"US"`, `"BR"` |
| `platform` | `platform` | `"ios"`, `"android"` |
| `app_version` | `app_version` | `"1.87.0+2026070117"` |

token 查表逻辑与训练时 `build_features.py` 完全一致：
- 空字符串 / None → `OOV (id=1)`
- 词表中存在 → 对应 int_id
- 词表中不存在（新版本号等）→ `OOV (id=1)`

每次请求日志示例：
```
encode  country_code='BR'(hit:3)  platform='android'(hit:2)  app_version='1.87.0+2026070117'(hit:3)
```
括号内为 vocab index，`hit` 表示命中，`OOV` 表示回退。

---

## Item Vector 内存管理（item_store.py）

服务启动时从 GCS 下载并常驻内存：

| 对象 | 来源文件 | 典型大小（~220 万游戏）|
|---|---|---|
| `item_vecs` | `item_vectors.npy` shape=(N, 64) float32 | ~537 MB |
| `game_id_to_idx` | `game_ids.json` | ~80 MB |
| model weights (user tower) | `two_tower_lite_best.pt` | < 1 MB |
| vocab (user 侧) | `vocab.json` | < 1 MB |

**常驻内存合计约 620 MB**，pod 配置 request 2Gi / limit 4Gi，为每日刷新时双 bundle 共存留足余量。

打分日志示例（三行一组）：
```
encode    country_code='BR'(hit:3)  platform='android'(hit:2)  app_version='1.87.0+2026070117'(hit:3)
user_emb  norm=0.6550  first8=-0.060 0.138 0.048 -0.128 0.030 -0.077 -0.185 -0.067
score     req=2081 hit=2081 hit_rate=1.000 mean_score=0.4887  user_id=21249658 country=BR platform=android app_version=1.87.0+2026070117
```

- `norm`：user embedding 的 L2 范数，可监控是否退化为零向量
- `hit_rate`：候选集中有 item vector 的比例，miss 通常为新游戏或已下架
- `mean_score`：命中游戏的平均预估分，可监控模型分布是否漂移

---

## 每日模型刷新

item vector 每日由 `export_item_vectors.py`（CronJob 05:00 UTC）更新至 GCS：

```
gs://rezona-ml/two_tower_lite/item_vectors/YYYY-MM-DD/
    item_vectors.npy
    game_ids.json
```

服务内置后台刷新线程（`_refresh_loop`）：
- 每天 **06:00 UTC**（北京时间 14:00）触发检查
- 启动时通过 `_find_ready_date()` 找最新的、ckpt + item_vecs **同时就绪**的日期，若当天 export 未完成自动回退到前一天
- 就绪后下载新 bundle，**原子替换** `_bundle` 引用（Python GIL 保证引用赋值原子性）
- 刷新期间正在处理的请求持有旧 bundle 引用，不受影响
- 若文件未就绪，每 5 分钟重试一次，最多重试 12 次（1 小时）

**内存安全**：`np.load(...).astype(np.float32, copy=False)` 避免刷新时产生额外副本，峰值内存控制在 ~1.35 GB（旧 bundle + 新 bundle），不超过 4Gi 上限。

---

## 优雅停机

服务注册了 `SIGTERM` handler，Kubernetes 滚动更新时：
1. 收到 SIGTERM → `server.stop(grace=20)` 停止接收新请求
2. 已有请求最多 20 秒内完成
3. 进程正常退出

`deployment.yaml` 中 `terminationGracePeriodSeconds: 30` 与 grace 20s 匹配，留有缓冲。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GCS_BUCKET` | **必填** | GCS bucket 名称，e.g. `rezona-ml` |
| `GCS_ARTIFACTS_PREFIX` | `two_tower_lite/artifacts` | 读取 `config.json` / `vocab.json` |
| `GCS_CKPT_PREFIX` | `two_tower_lite/ckpts` | 读取 `two_tower_lite_best.pt` |
| `GCS_ITEM_VECTORS_PREFIX` | `two_tower_lite/item_vectors` | 读取 `item_vectors.npy` / `game_ids.json` |
| `GRPC_PORT` | `50051` | 监听端口 |
| `GRPC_WORKERS` | `os.cpu_count() × 2` | gRPC 线程池大小，numpy BLAS 释放 GIL 可真正并行 |
| `REFRESH_ENABLED` | `true` | 是否启用每日自动刷新 |

**本地开发**（`GCS_BUCKET` 不填时走本地路径）：

| 变量 | 说明 |
|---|---|
| `ART_DIR` | artifacts 目录（含 config.json、vocab.json）|
| `CKPT_PATH` | checkpoint 文件路径 |
| `ITEM_VECTORS_DIR` | 含 item_vectors.npy 和 game_ids.json 的目录 |

---

## 部署

### 编译镜像

```bash
# macOS 需指定 Python 3.10+（gcloud 依赖）
export CLOUDSDK_PYTHON=$(brew --prefix python@3.11)/bin/python3.11

gcloud builds submit \
  --tag us-east1-docker.pkg.dev/rezonaai/reco-model/ml-trainer:latest \
  --machine-type=e2-highcpu-8 \
  .
```

Dockerfile 在 `COPY serving/` 后自动执行 `grpc_tools.protoc`，无需手动生成 pb2 文件。

### 部署 / 更新服务

```bash
kubectl apply -f serving/deployment.yaml
```

### 滚动重启（镜像更新后）

```bash
kubectl rollout restart deployment/two-tower-lite-ranker -n production
kubectl rollout status deployment/two-tower-lite-ranker -n production
```

### 查看日志

```bash
kubectl logs -f -n production -l app=two-tower-lite-ranker
```

---

## 集群内访问

服务类型为 ClusterIP，集群内其他 Pod 通过以下地址访问：

```
two-tower-lite-ranker.production.svc.cluster.local:50051
```

---

## 本地测试

使用 `grpcurl`（需安装：`brew install grpcurl`）：

```bash
# 先做端口转发
kubectl port-forward svc/two-tower-lite-ranker 50051:50051 -n production

# 另开终端发请求（game_ids 为 int64）
grpcurl -plaintext \
  -import-path serving/ \
  -proto twotower_ranker.proto \
  -d '{
    "user": {
      "user_id":      "u_12345",
      "country_code": "CN",
      "platform":     "android",
      "app_version":  "1.87.0+2026070117"
    },
    "game_ids": [5807787, 7826766, 9764922, 6621248, 8559735]
  }' \
  localhost:50051 \
  twotower_ranker.TwoTowerRankerService/Score
```

预期响应：
```json
{
  "scores": [
    { "gameId": "5807787", "score": 0.78 },
    ...
  ],
  "modelDate": "2026-07-09"
}
```

> `gameId` 在 JSON 中以字符串形式展示（JSON 标准对 int64 的处理方式），proto wire 层实际为 int64。
