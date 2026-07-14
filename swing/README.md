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

从每条曝光记录的 `recent_games` 字段中提取用户的正向交互：

- `interaction_label > 0`：有显式互动（点赞 / 关注 / 分享 / 评论 / Remix），**或**
- `play_time > 16s`：有效播放（`recent_games[].play_time` 单位为秒）

### 交互权重

每条正样本赋予权重 `w = label_score × play_time_bonus`，权重用于在 Pass 1 排序时选取每个用户的 Top 游戏（`USER_GAME_CAP`），**不参与最终相似度计算**。

**label_score**（基础 1.0，叠加各行为）：

| 行为 | 加分 |
|---|---|
| 点赞 | +1.50 |
| 分享 | +1.00 |
| 评论 | +3.00 |
| 关注 | +1.00 |
| Remix | +1.00 |

**play_time_bonus**（对数缩放，1.0× @ 60s，上限 2.0×；无 play_time 时 floor 0.5）：

```
pt_bonus = min(log1p(play_time) / log1p(60), 2.0)
```

### Swing 相似度

#### 公式

$$\text{sim}(i,j) = \sum_{u \in U_i \cap U_j} \sum_{v \in U_i \cap U_j} \frac{1}{\alpha + |I_u \cap I_v|}$$

符号含义：
- `U_i` — 与游戏 i 有正向交互的用户集合
- `U_i ∩ U_j` — 同时交互过游戏 i 和 j 的共现用户集合（co-users）
- `I_u` — 用户 u 交互过的游戏集合
- `|I_u ∩ I_v|` — 用户 u 和 v 共同交互过的游戏数量

**外层双重求和**：枚举共现用户集合中所有有序用户对 (u, v)（含 u=v 自配对），共现用户越多、用户对越多，sim 随 co_users 近似二次增长。

**每对用户的贡献 `1 / (α + |I_u ∩ I_v|)`**：
- 口味高度重叠的两人（`|I_u ∩ I_v|` 大）→ 分母大 → 贡献小（信号弱，可能是用户泡沫）
- 口味差异大的两人（`|I_u ∩ I_v|` 小）→ 分母趋近 α → 贡献接近 `1/α`（强相似信号）

核心思想：**奖励「口味差异大的用户群都共同选择了 i 和 j」，过滤用户泡沫带来的虚假相似度。**

> **关于 popularity 归一化**：原版公式不含 `sqrt(|U_i| × |U_j|)` 显式归一化，依赖隐式抑制——热门游戏的共现用户彼此口味相近，`|I_u ∩ I_v|` 较大，每项贡献自然被压低，从而部分对冲了共现用户数量多带来的偏差。

#### 实现（两阶段）

**Pass 1**：收集每个用户的正向游戏集合，按权重排序取 Top-`USER_GAME_CAP`，构建 `user → {game_id: weight}`。

**Pass 2**：用户遍历，为每个游戏对 `(A, B)` 记录共现用户列表（capped at `MAX_PAIR_USERS`）：

```
co_occ[(A, B)].co_users += 1
co_occ[(A, B)].uids.append(uid)
```

**Step 3**：对每个游戏对 (A, B) 遍历 uid 列表的所有用户对，计算最终相似度：

```
score(A, B) = Σ_{u,v ∈ uids}  1 / (α + |I_u ∩ I_v|)   (u=v 贡献1次，u≠v 贡献2次)
```

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

## 实验记录

评估方法：时间切割 Recall@K——用前 17 天数据建索引，第 18 天（eval_day）作为测试集。测试样本：用户在 eval_day 有正向 target（explicit label > 0 或 playing_time > 10s），且历史中存在至少一个 trigger 游戏。

| 版本 | 正样本 | qualified items | avg neighbors | Recall@10 | Recall@20 | Recall@50 | 耗时 |
|---|---|---|---|---|---|---|---|
| 简化 Swing（仅 explicit label>0） | 22.2M | 11,112 | 2.7 | 1.09% | 1.12% | 1.12% | 194s |
| 简化 Swing（+ effective play >16s） | 58.7M | 26,026 | 4.2 | 1.85% | 1.98% | 1.98% | 194s |
| **原版 Swing（当前生产版本）** | 58.7M | 26,026 | 24.7 | **10.1%** | **15.8%** | **25.2%** | 925s |

**关键发现**：

- 简化版 Swing 存在饱和问题：`sum_w ∝ co_users`，导致分子分母同步增长，高共现对无法获得应有的更高分数，avg neighbors 极低（2.7–4.2），Recall@50 与 Recall@10 几乎相同（说明排名无意义）。
- 原版 Swing 的双重求和结构使 sim 随 co_users 二次增长，avg neighbors 从 4.2 → 24.7，Recall@50 从 1.98% → 25.2%（12.7× 提升）。
- 增加 effective play（>16s）作为正样本大幅扩展 qualified items（11k → 26k），提升索引覆盖度，是两个改进中更容易实现的一步。

---

## 本地开发（离线数据）

```bash
# 使用本地 ndjson（原始格式）
DATA_DIR=/path/to/ndjson OUT_DIR=/path/to/output python swing/build_i2i.py
```

`build_i2i.py` 直接读取完整的 `reco_labeled_log_flat_YYYY-MM-DD_realshow.ndjson` 文件，无需 BQ/GCS，适合本地调试算法参数。
