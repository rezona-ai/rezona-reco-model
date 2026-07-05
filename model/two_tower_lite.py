#!/usr/bin/env python3
"""
Lightweight coarse-ranking two-tower (production Config A).

User tower : context only — country / platform / app_version  (NO user_id, NO behavior
             sequence, NO multihot). This genuinely drops the 13.5M-param user_id table.
Item tower : game_id / author_id / category / tag / art_style + 16 numeric stats (same as v2).

MLPs are plain Linear+ReLU (no BatchNorm, no Dropout); head is a plain dot product (no bias).

Rationale (Exp9): a zero-personalization user tower keeps ~96% of AUC / ~99% of GAUC and is
BEST for the 64.5% new users, at a fraction of the params and with no user-sequence/ID infra.
"""
import torch
import torch.nn as nn

EMB_DIM = 32
TOWER_OUT = 64
PAD = 0
CONTEXT_FEATS = {"country_code", "platform", "app_version"}


class TwoTowerLite(nn.Module):
    def __init__(self, cfg, dropout=0.0):
        super().__init__()
        self.cfg = cfg
        sizes = cfg["vocab_sizes"]
        self.user_cats = [c for c in cfg["cat_single"]
                          if c["tower"] == "user" and c["feat"] in CONTEXT_FEATS]
        self.item_cats = [c for c in cfg["cat_single"] if c["tower"] == "item"]
        self.n_num = len(cfg["numeric"]["features"])

        # only build the embedding tables this model actually uses (real param savings)
        used = {c["vocab"] for c in self.user_cats} | {c["vocab"] for c in self.item_cats}
        self.emb = nn.ModuleDict({v: nn.Embedding(sizes[v], EMB_DIM, padding_idx=PAD) for v in used})
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.05)
            with torch.no_grad():
                e.weight[PAD].zero_()

        # plain MLPs: Linear + ReLU only (no BatchNorm, no Dropout)
        u_in = len(self.user_cats) * EMB_DIM
        self.user_mlp = nn.Sequential(
            nn.Linear(u_in, 128), nn.ReLU(),
            nn.Linear(128, TOWER_OUT),
        )
        self.num_mlp = nn.Sequential(nn.Linear(self.n_num, 64), nn.ReLU(), nn.Linear(64, EMB_DIM))
        i_in = len(self.item_cats) * EMB_DIM + EMB_DIM
        self.item_mlp = nn.Sequential(
            nn.Linear(i_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, TOWER_OUT),
        )

    def user_vec(self, b):
        parts = [self.emb[c["vocab"]](b[f"cat__{c['feat']}"]) for c in self.user_cats]
        return self.user_mlp(torch.cat(parts, dim=-1))

    def item_vec(self, b):
        parts = [self.emb[c["vocab"]](b[f"cat__{c['feat']}"]) for c in self.item_cats]
        parts.append(self.num_mlp(b["num"]))
        return self.item_mlp(torch.cat(parts, dim=-1))

    def forward(self, b):
        # plain dot product, no bias term
        return (self.user_vec(b) * self.item_vec(b)).sum(-1)
