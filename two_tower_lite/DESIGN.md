# 粗排模型设计

## 1. 定位与职责

粗排（Coarse Ranking）处于召回与精排之间，负责将召回层输出的数千量级候选集快速压缩为高质量子集。核心约束：**在极低延迟内完成全量候选打分**，同时保证足够的排序区分度，使精排损失的正样本尽量少。

```
召回层（Swing i2i / 热门池 / 关注链 / 创作者扶持 …）
          ↓  数千候选
        粗排（TwoTowerLite）        ← 本模块
          ↓  压缩后子集
        精排 / 重排
          ↓
        最终结果
```

---

## 2. 为什么用双塔

双塔（Two-Tower）是粗排的标准范式：item 向量离线预计算，在线只需计算 user 向量并做点积，时间复杂度 O(d) 与候选集大小无关。相比 cross-attention 类精排模型，双塔牺牲了 user-item 交叉信息，但延迟可控，适合粗排量级。

---

## 3. User 塔为什么只用 Context

离线分析结论：
- 测试集 **74% user_id** 在训练集中未出现（OOV）
- **57% 请求**无任何历史行为记录（recent_games 为空）

user_id Embedding 在如此高 OOV 率下泛化极差。消融实验（Exp9）表明，context-only 方案仅损失 AUC 约 0.024，且对新用户效果反而更优。User 塔放弃 user_id 和行为序列，以极低参数量换取**冷启动鲁棒性**。

---

## 4. 模型架构

```
User 塔（context-only）              Item 塔
  country  (Emb 32)                game_id   (Emb 32)
  platform (Emb 32)                author_id (Emb 32)
  app_ver  (Emb 32)                category  (Emb 32)
      │                            tag       (Emb 32)
      │                            art_style (Emb 32)
      │                            15维数值统计
      │                               └─ num_mlp(15→64→32)
      │                                    │
   concat(96)                         concat(192)
      │                                    │
   MLP(96→128→64)                   MLP(192→256→128→64)
      │                                    │
   user_vec(64)              ──dot──  item_vec(64)
                                          │
                                       logit
                                          │
                                   BCEWithLogitsLoss
```

总参数量约 **0.73M**，主要集中在 item 侧 Embedding 表。轻量设计降低过拟合风险（训练/测试 AUC gap < 0.005）。

相关性计算无偏置项，logit 直接用于排序：

$$\text{logit}(u, i) = \vec{u} \cdot \vec{i} = \sum_{k=1}^{64} u_k \cdot i_k$$

---

## 5. 标签设计

**有效播放二分类**：

$$\text{label} = \mathbb{1}[\text{play\_time} > 33430 \text{ ms}]$$

阈值 33.43 秒对应用户完成基本内容浏览的最低时长，正例率约 **10%**。

不用显式互动行为（like/follow/share）的原因：在数据中占比均低于 1%，正例极稀疏，不适合作为主训练目标；多任务扩展时可作为辅助信号引入。

---

## 6. 推理设计

采用**离线预计算 + 在线点积**模式，将 item 侧计算完全从请求链路中剥离：

```
离线（每日更新）:
  所有已发布游戏 → Item Tower → item_vectors (N×64, float32)

在线（每次请求）:
  请求上下文 → User Tower → user_vec (64,)
  scores = sigmoid(item_vecs[candidates] @ user_vec)
```

服务维护 `ServingBundle` 原子快照（model + encoder + item_vecs），每日热加载无需重启。**artifacts / ckpt / item_vectors 必须来自同一天**，否则 vocab 尺寸不匹配导致崩溃。
