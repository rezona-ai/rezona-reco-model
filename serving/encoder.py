"""
User-side feature encoder for online serving.
Must stay in sync with build_features.py / two_tower_lite.py.
"""
import logging

import torch

log = logging.getLogger(__name__)

PAD, OOV = 0, 1
CONTEXT_FEATS = {"country_code", "platform", "app_version"}


def _norm_token(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return str(v)


class UserEncoder:
    def __init__(self, cfg: dict, vocab: dict):
        # mirrors TwoTowerLite.__init__: user tower = context feats only
        self.user_cats = [
            c for c in cfg["cat_single"]
            if c["tower"] == "user" and c["feat"] in CONTEXT_FEATS
        ]
        self.vocab = vocab  # {vocab_key: {token: int_id}}

    def encode(self, country_code: str, platform: str, app_version: str) -> dict:
        """
        Returns {cat__<feat>: LongTensor([id])} ready for model.user_vec().
        Unknown tokens → OOV (index 1), same as training.
        """
        feat_to_val = {
            "country_code": country_code,
            "platform":     platform,
            "app_version":  app_version,
        }
        batch = {}
        parts = []
        for c in self.user_cats:
            raw = feat_to_val.get(c["feat"])
            t   = _norm_token(raw)
            hit = t is not None and t in self.vocab.get(c["vocab"], {})
            idx = self.vocab.get(c["vocab"], {}).get(t, OOV) if hit else OOV
            batch[f"cat__{c['feat']}"] = torch.tensor([idx], dtype=torch.long)
            parts.append(f"{c['feat']}={t!r}({'hit' if hit else 'OOV'}:{idx})")
        log.info("encode  %s", "  ".join(parts))
        return batch
