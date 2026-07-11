"""
ServingBundle — model + encoder + item vectors 的原子快照。

server 持有一个 _bundle 引用，刷新时直接替换整个引用。
Python attribute assignment 在 GIL 下是原子操作，不需要额外锁。
"""
import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class ServingBundle:
    def __init__(self, model, encoder, item_vecs: np.ndarray,
                 game_id_to_idx: dict, date: str):
        self._model   = model          # TwoTowerLite，eval 模式
        self._encoder = encoder        # UserEncoder
        self._vecs    = item_vecs      # (N, 64) float32
        self._mapping = game_id_to_idx # game_id str → row index int
        self.date     = date

    def score(self, user_id: str, country_code: str, platform: str, app_version: str,
              game_ids: list[int]) -> np.ndarray:
        """
        返回 float32 数组，长度 = len(game_ids)，顺序与输入一致。
        不在 item_vecs 里的 game_id → 0.0。
        """
        batch = self._encoder.encode(country_code, platform, app_version)
        with torch.no_grad():
            user_vec = self._model.user_vec(batch).numpy()  # (1, 64)
        log.info(
            "user_emb  norm=%.4f  first8=%s",
            float(np.linalg.norm(user_vec)),
            " ".join(f"{v:.3f}" for v in user_vec[0, :8]),
        )

        n_req   = len(game_ids)
        # list comprehension is faster than explicit append loop under GIL
        hits    = [(pos, self._mapping[gid]) for pos, gid in enumerate(game_ids)
                   if gid in self._mapping]
        scores  = np.zeros(n_req, dtype=np.float32)
        n_hit   = len(hits)
        if hits:
            positions, idxs = zip(*hits)
            vecs   = self._vecs[list(idxs)]                        # (K, 64)
            logits = (vecs @ user_vec.T).squeeze(-1)               # (K,) — releases GIL
            s      = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
            scores[list(positions)] = np.atleast_1d(s)             # vectorized write, no loop

        hit_rate  = n_hit / n_req if n_req else 0.0
        mean_score = float(scores[list(positions)].mean()) if n_hit else 0.0
        log.info(
            "score  req=%d hit=%d hit_rate=%.3f mean_score=%.4f  "
            "user_id=%s country=%s platform=%s app_version=%s",
            n_req, n_hit, hit_rate, mean_score,
            user_id, country_code, platform, app_version,
        )

        return scores
