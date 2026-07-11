# Rezona 粗排在线打分服务 — 设计文档

## 1. 定位

粗排阶段：从召回集（数百到数千个候选 game）中实时打分排序，供后续精排使用。

```
召回服务  ──候选 game_ids──▶  [本服务]  ──(game_id, score)──▶  精排 / 截断
              user context ──────────────▶
```

---

## 2. 架构概览

```
                  ┌─────────────────────────────────────────┐
  gRPC client     │           RankerServer                  │
                  │                                         │
  ScoreRequest    │  1. encode user features                │
  ─────────────▶  │     → user_vec (64-dim)                 │
                  │                                         │
                  │  2. lookup pre-computed item_vecs       │
                  │     for candidate game_ids              │
                  │                                         │
                  │  3. dot product + sigmoid → scores      │
  ◀─────────────  │                                         │
  ScoreResponse   │  [in-memory]                            │
                  │  item_vecs: np.ndarray (N, 64)          │
                  │  game_id→idx: dict                      │
                  │  model (user_tower only)                 │
                  └─────────────────────────────────────────┘
```

**核心思路**：item vector 离线预计算（`export_item_vectors.py` 每日跑一次），
服务启动时加载进内存。每次请求只需前向一次 user tower + 向量点积，无 BQ/GCS 依赖。

---

## 3. Proto 接口

```protobuf
syntax = "proto3";
package rezona.ranker.v1;

// ── request ──────────────────────────────────────────────
message UserFeatures {
  string country_code = 1;   // e.g. "US"
  string platform     = 2;   // e.g. "ios"
  string app_version  = 3;   // e.g. "2.1.0"
}

message ScoreRequest {
  UserFeatures user          = 1;
  repeated string game_ids   = 2;   // 候选集，顺序不影响结果
}

// ── response ─────────────────────────────────────────────
message GameScore {
  string game_id = 1;
  float  score   = 2;   // sigmoid 后的预估播放概率 [0, 1]
}

message ScoreResponse {
  repeated GameScore scores  = 1;   // 与请求 game_ids 顺序对应
  string             model_date = 2; // item vectors 来自哪天，方便 debug
}

// ── service ──────────────────────────────────────────────
service RankerService {
  rpc Score(ScoreRequest) returns (ScoreResponse);
}
```

**设计选择说明**

- 只接受 user context（country/platform/app_version），与 TwoTowerLite user tower 完全对齐（无 user_id，无行为序列）。
- 返回 sigmoid 概率而非原始 logit，调用方可直接用作排序分。
- `model_date` 方便排查线上打分与离线评估对不齐的问题。

---

## 4. 请求处理流程

```
ScoreRequest
    │
    ▼
encode_user(user_features)
    │  ① 查 vocab: country/platform/app_version → token id (OOV→1)
    │  ② model.user_vec(batch)  — 小 MLP, ~0.1ms
    ▼
user_vec: (1, 64) float32
    │
    ▼
lookup_item_vecs(game_ids)
    │  ① game_id → idx (dict, O(1) per id)
    │  ② item_vecs[idxs] — numpy fancy indexing
    │  ③ 不在词表的 game_id → 使用 OOV item vec（预先备好）
    ▼
item_vecs: (K, 64) float32
    │
    ▼
scores = sigmoid(user_vec @ item_vecs.T)  — (K,) float32
    │
    ▼
ScoreResponse(scores=[GameScore(game_ids[i], scores[i]) ...])
```

---

## 5. 内存布局

| 对象 | 来源 | 大小（~4k games） |
|---|---|---|
| `item_vecs` | `item_vectors.npy` (N, 64) float32 | ~1 MB |
| `game_id_to_idx` | `game_ids.json` | < 0.5 MB |
| `oov_item_vec` | `item_vecs[1]` (OOV行) | 忽略 |
| model weights | `two_tower_lite_best.pt` (user tower only) | < 1 MB |
| vocab (user侧) | `vocab.json` (country/platform/app_ver) | < 0.1 MB |

总内存 << 10 MB，无需分片或外部 kv store。

---

## 6. 模型刷新策略

item vectors 每日由 `export_item_vectors.py` 更新到 GCS：
```
gs://{GCS_BUCKET}/two_tower_lite/item_vectors/{YYYY-MM-DD}/
```

**两种方案：**

### 方案 A — 定时热加载（推荐）
服务内启一个后台线程，每小时检查 GCS 最新日期目录，有新版本则下载并原子替换内存中的 `item_vecs` 和 `game_id_to_idx`（加读写锁）。

优点：无需重启 pod，零停机刷新。  
缺点：实现稍复杂（需 threading.Lock 或 asyncio.Lock）。

### 方案 B — 重启加载
每日 item vector 更新后触发 pod rolling restart，启动时从 GCS 拉最新版本。

优点：实现最简单。  
缺点：每次刷新有短暂服务中断（rolling restart 期间）。

---

## 7. 服务部署

```
serving/
├── DESIGN.md              # 本文档
├── ranker.proto           # proto 定义
├── ranker_pb2.py          # generated
├── ranker_pb2_grpc.py     # generated
├── server.py              # gRPC server 主体
├── encoder.py             # 特征编码（与 build_features.py 对齐）
├── item_store.py          # item vector 内存管理 + 刷新
└── Dockerfile             # 单独镜像或复用 ml-trainer
```

**环境变量**
```
GCS_BUCKET
GCS_ARTIFACTS_PREFIX   # 读 config.json / vocab.json
GCS_CKPT_PREFIX        # 读 model ckpt
GCS_ITEM_VECTORS_PREFIX
GRPC_PORT              # default 50051
RELOAD_INTERVAL_SEC    # default 3600，0=禁用自动刷新
```

---

## 8. 待确认问题

1. ~~**候选集规模**~~ → 3k–4k，服务只返回分数不做截断。✓
2. **延迟目标**：p99 要求多少 ms？（当前方案预估 < 5ms，无 GPU）
3. **模型刷新频率**：item vectors 每日更新，user tower ckpt 多久更新一次？两者是否同步刷新？
4. ~~**OOV game 处理**~~ → score = 0.0。✓
5. **认证**：gRPC 是否需要 mTLS / token？
