# Swing i2i 召回索引

## 定位

基于 Swing 算法构建 item-to-item 相似索引，用于召回阶段：给定用户最近交互的游戏，找出相似游戏作为候选集。

```
用户历史行为（recent_games）
        │
        ▼
  [本模块] Swing i2i
        │
        ▼
  game_id → Top-50 相似游戏（带分数）
```

---

## 目录结构

```
swing/
├── run.py              # 云端入口：BQ 拉取 → slim 缓存 → Swing → 上传 GCS
├── build_i2i.py        # Swing 核心算法（可独立本地运行）
├── cronjob_swing.yaml  # GKE CronJob 定义
└── README.md
```

---

## 算法说明

### 正样本定义

从每条曝光记录的 `recent_games` 字段中提取用户的正向交互，满足以下任一条件即为正样本：

- `interaction_label > 0`：有显式互动（点赞 / 关注 / 分享 / 评论 / Remix）
- `play_time > 100s`：长播放（隐式正样本，约 p80 分位）

### 交互权重

每条正样本赋予权重 `w = label_score × play_time_bonus`：

**label_score**（基础 1.0，叠加各行为）：

| 行为 | 加分 |
|---|---|
| 仅长播放（label=0） | 0.8（低于显式互动） |
| 点赞 | +0.50 |
| 分享 | +0.50 |
| 评论 | +0.70 |
| 关注 | +1.00 |
| Remix | +1.50 |

**play_time_bonus**（对数缩放，1.0× @ 60s，上限 2.0×）：

```
pt_bonus = min(log1p(play_time) / log1p(60), 2.0)
```

### Swing 相似度

**Pass 1**：收集每个用户的正向游戏集合，按权重取 Top-50（`USER_GAME_CAP`），构建 `user → {game_id: weight}`。

**Pass 2**：对每个用户枚举其正向游戏中的所有对 `(A, B)`，累加共现分：

```
raw_score(A, B) += w(A) × w(B)
co_users(A, B)  += 1
```

**Step 3**：计算最终相似度：

```
sim(A, B) = raw_score(A, B) / (α + co_users(A, B)) / sqrt(n_users(A) × n_users(B))
```

- **α（ALPHA=5）**：平滑项，压制共现用户数少的噪声对
- **几何平均归一化**：消除热门游戏的 popularity bias
- **MAX_PAIR_USERS=200**：截断共现用户上限，防止超热门对主导排名

每个游戏保留相似度最高的 Top-50 邻居（`MIN_CO_USERS=2` 过滤极稀疏对）。

---

## 数据流（云端）

```
BQ reco_labeled_log_flat
        │  每天新增一天（SELECT uid + recent_games，仅两列）
        ▼
gs://rezona-ml/swing/slim/YYYY-MM-DD.ndjson.gz     ← 18 天滚动缓存
        │  每行：{"uid": int, "games": [{"gid": int, "label": int, "pt": int}]}
        ▼
  Swing 算法（Pass 1 → Pass 2 → Step 3）
        ▼
gs://rezona-ml/swing/artifacts/YYYY-MM-DD/         ← 每天存档
gs://rezona-ml/swing/artifacts/latest/             ← 最新版本（供线上读取）
    ├── i2i_index.json          game_id → [{game_id, score, co_users}, ...]
    ├── i2i_index_simple.json   game_id → [game_id, ...]
    └── i2i_stats.json          构建统计
```

slim 数据与粗排（`two_tower_lite/slim/`）完全分开，互不干扰。

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `SWING_DAYS` | 18 | 使用最近 N 天数据 |
| `ALPHA` | 5.0 | Swing 平滑，越大越压制小共现对 |
| `MIN_ITEM_USERS` | 3 | 游戏至少被 N 个用户正向交互才进入索引 |
| `MIN_CO_USERS` | 2 | 游戏对至少共现 N 个用户才保留 |
| `TOP_K` | 50 | 每个游戏保留的邻居数 |
| `MAX_PAIR_USERS` | 200 | 共现用户数上限（防热门对膨胀） |
| `USER_GAME_CAP` | 50 | 每个用户最多贡献的正向游戏数 |
| `LONG_PLAY_SEC` | 100 | 隐式正样本的播放时长阈值（秒） |

---

## 上线流程

### 首次上线

```bash
# 1. 编译共用镜像（根目录执行）
./deploy.sh

# 2. 手动触发一次完整运行（首次会拉取 18 天历史 BQ 数据，约 2 小时）
DATE=$(date +%m%d%H%M)
kubectl create job --from=cronjob/swing-i2i-build "swing-now-${DATE}" -n production

# 3. 跟踪日志
kubectl logs -f job/swing-now-${DATE} -n production

# 4. 确认产物已上传
gsutil ls gs://rezona-ml/swing/artifacts/latest/
```

### 日常运行

CronJob `swing-i2i-build` 每天 **04:30 UTC**（北京时间 12:30）自动执行：
1. 检查 slim 缓存，仅拉取昨天新数据（约 5～10 分钟）
2. 下载 18 天 slim 文件，运行 Swing 算法（约 20～40 分钟）
3. 上传结果到 `gs://rezona-ml/swing/artifacts/latest/`

### 查看状态

```bash
# CronJob 状态
kubectl get cronjob swing-i2i-build -n production

# 最近一次 job 日志
kubectl logs -n production -l job-name=<job-name>

# 确认 GCS 产物
gsutil ls gs://rezona-ml/swing/artifacts/latest/
gsutil cat gs://rezona-ml/swing/artifacts/latest/i2i_stats.json
```

### 手动触发（按需重跑）

```bash
DATE=$(date +%m%d%H%M)
kubectl create job --from=cronjob/swing-i2i-build "swing-now-${DATE}" -n production
kubectl logs -f job/swing-now-${DATE} -n production
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GCS_BUCKET` | **必填** | e.g. `rezona-ml` |
| `GCS_SLIM_PREFIX` | `swing/slim` | slim 缓存路径 |
| `GCS_ARTIFACTS_PREFIX` | `swing/artifacts` | 产物输出路径 |
| `BQ_PROJECT` | `rezonaai` | BigQuery 项目 |
| `BQ_TABLE` | `rezonaai.datalake.reco_labeled_log_flat` | 数据源表 |
| `BQ_WHERE` | `context_info.realshow = TRUE` | 过滤条件 |
| `SWING_DAYS` | `18` | 滚动窗口天数 |

---

## 本地开发（离线数据）

```bash
# 使用本地 ndjson（原始格式）
DATA_DIR=/path/to/ndjson OUT_DIR=/path/to/output python swing/build_i2i.py
```

`build_i2i.py` 直接读取完整的 `reco_labeled_log_flat_YYYY-MM-DD_realshow.ndjson` 文件，无需 BQ/GCS，适合本地调试算法参数。
