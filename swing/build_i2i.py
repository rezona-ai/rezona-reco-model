#!/usr/bin/env python3
"""
i2i (item-to-item) recall index builder.

Input : data/reco_labeled_log_flat_YYYY-MM-DD_realshow.ndjson (18 days)
Output: artifacts/i2i_index.json
            game_id -> [{"game_id": gid, "score": float, "co_users": int}, ...]
        artifacts/i2i_stats.json   build statistics

Algorithm: Swing co-occurrence similarity
-----------------------------------------
For each user, collect positively-interacted games from recent_games.
For every pair (A, B) a user engaged with, accumulate:

    raw_score(A, B) += w_u(A) * w_u(B) / (ALPHA + |N(A,B)|)

Then normalize by geometric-mean of per-item user counts:

    sim(A, B) = raw_score(A, B) / sqrt(n_users(A) * n_users(B))

Interaction label bitmask:
    bit 0 (1)  = 点赞 (like)
    bit 1 (2)  = 关注 (follow)
    bit 2 (4)  = 分享 (share)
    bit 3 (8)  = 评论 (comment)
    bit 4 (16) = Remix

Positive threshold: label > 0 (explicit interaction only).
Weight = label_strength * play_time_bonus.
"""
import gzip, os, sys, json, math, time
from collections import defaultdict

import orjson

DATA_DIR  = os.environ.get("DATA_DIR",  os.path.join(os.path.dirname(__file__), "..", "data"))
OUT_DIR   = os.environ.get("OUT_DIR",   os.path.join(os.path.dirname(__file__), "..", "artifacts"))
FILE_TMPL = "reco_labeled_log_flat_{day}_realshow.ndjson"

# ---------------------------------------------------------------------------
# build_user_game_map variant for swing slim format
#   slim line: {"uid": int, "games": [{"gid": int, "label": int, "pt": int}, ...]}
# ---------------------------------------------------------------------------
def build_user_game_map_from_slim(paths: list) -> tuple:
    """Read pre-extracted swing slim files instead of full ndjson."""
    user_games: dict = defaultdict(dict)
    n_records = 0
    n_pos_interactions = 0

    for path in paths:
        open_fn = gzip.open if str(path).endswith(".gz") else open
        with open_fn(path, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                uid = rec.get("uid")
                if not uid:
                    continue
                uid = int(uid)
                n_records += 1
                ug = user_games[uid]
                for g in rec.get("games") or []:
                    gid = g.get("gid")
                    if not gid:
                        continue
                    gid = int(gid)
                    w = interaction_weight(int(g.get("label") or 0), int(g.get("pt") or 0))
                    if w > 0:
                        n_pos_interactions += 1
                        if w > ug.get(gid, 0.0):
                            ug[gid] = w

    return dict(user_games), n_records, n_pos_interactions

# --- tunables ---------------------------------------------------------------
MIN_ITEM_USERS = 3     # min distinct users for a game to enter the index
MIN_CO_USERS   = 2     # min co-occurring users to keep a pair
TOP_K          = 50    # neighbors to keep per game
ALPHA          = 5.0   # Swing smoothing: penalises trivially popular pairs
MAX_PAIR_USERS = 200   # cap co-user count to limit dominant-pair inflation
USER_GAME_CAP  = 50    # max positive games per user (clip to most-engaged)


# ---------------------------------------------------------------------------
# step 1: weight of a single interaction (game in recent_games)
# ---------------------------------------------------------------------------
BIT_LIKE    = 1 << 0   # 1  点赞
BIT_FOLLOW  = 1 << 1   # 2  关注
BIT_SHARE   = 1 << 2   # 4  分享
BIT_COMMENT = 1 << 3   # 8  评论
BIT_REMIX   = 1 << 4   # 16 Remix

EFFECTIVE_PLAY_S = 16   # recent_games play_time unit is seconds

def interaction_weight(label: int, play_time: int) -> float:
    # positive gate: explicit interaction (label > 0) OR effective play (play_time > 16s)
    if label == 0 and play_time <= EFFECTIVE_PLAY_S:
        return 0.0

    has_like    = bool(label & BIT_LIKE)
    has_follow  = bool(label & BIT_FOLLOW)
    has_share   = bool(label & BIT_SHARE)
    has_comment = bool(label & BIT_COMMENT)
    has_remix   = bool(label & BIT_REMIX)

    label_score = 1.0
    if has_like:    label_score += 1.50
    if has_share:   label_score += 1.00
    if has_comment: label_score += 3.00
    if has_follow:  label_score += 1.00
    if has_remix:   label_score += 1.00

    # play_time bonus: log-scale, 1.0x at 60s, capped at 2.0x; floor 0.5 if no play_time
    pt_bonus = min(math.log1p(play_time) / math.log1p(60), 2.0) if play_time > 0 else 0.5
    return label_score * pt_bonus


# ---------------------------------------------------------------------------
# pass 1: collect per-user positive game weights
#   user_games: user_id -> {game_id: max_weight}
# ---------------------------------------------------------------------------
def build_user_game_map(days: list[str]) -> dict:
    user_games: dict[int, dict[int, float]] = defaultdict(dict)
    n_records = 0
    n_pos_interactions = 0

    for day in days:
        path = os.path.join(DATA_DIR, FILE_TMPL.format(day=day))
        with open(path, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                n_records += 1
                uid = rec["user_info"].get("user_id")
                if not uid:
                    continue
                uid = int(uid)
                ug = user_games[uid]
                for g in rec["user_info"].get("recent_games") or []:
                    gid = g.get("game_id")
                    if not gid:
                        continue
                    gid = int(gid)
                    lbl = int(g.get("interaction_label") or 0)
                    pt  = int(g.get("play_time") or 0)
                    w   = interaction_weight(lbl, pt)
                    if w > 0:
                        n_pos_interactions += 1
                        if w > ug.get(gid, 0.0):
                            ug[gid] = w

        print(f"  {day}: {n_records:,} records so far, {n_pos_interactions:,} positive interactions")

    return dict(user_games), n_records, n_pos_interactions


# ---------------------------------------------------------------------------
# pass 2: collect co-occurring user lists per game pair
#
# Same user-traversal structure as simplified Swing (O(U × G²)),
# but stores uid list instead of sum_w — cheap, cache-friendly.
# The actual |I_u ∩ I_v| computation is deferred to build_index (pass 3),
# where it only runs over pairs that genuinely exist.
# ---------------------------------------------------------------------------
def build_co_occurrence(
    user_games: dict,
    min_item_users: int,
) -> tuple[dict, dict, dict]:
    # count distinct users per game
    item_user_cnt: dict[int, int] = defaultdict(int)
    for uid, games in user_games.items():
        for gid in games:
            item_user_cnt[gid] += 1

    qualified = {gid for gid, cnt in item_user_cnt.items() if cnt >= min_item_users}
    print(f"  qualified items (>={min_item_users} users): {len(qualified):,}")

    # co_occ: (min_gid, max_gid) -> [co_users, [uid, ...]]
    # uid list is capped at MAX_PAIR_USERS; used in build_index for |I_u ∩ I_v|
    co_occ: dict[tuple, list] = defaultdict(lambda: [0, []])

    n_users_processed = 0
    for uid, games in user_games.items():
        qual_games = {gid: w for gid, w in games.items() if gid in qualified}
        if len(qual_games) < 2:
            continue
        top_games = sorted(qual_games.items(), key=lambda kv: -kv[1])[:USER_GAME_CAP]
        n_users_processed += 1
        for i in range(len(top_games)):
            gid_a, _ = top_games[i]
            for j in range(i + 1, len(top_games)):
                gid_b, _ = top_games[j]
                key = (min(gid_a, gid_b), max(gid_a, gid_b))
                entry = co_occ[key]
                entry[0] += 1
                if entry[0] <= MAX_PAIR_USERS:
                    entry[1].append(uid)

    print(f"  users contributing pairs: {n_users_processed:,}")
    print(f"  total co-occurrence pairs: {len(co_occ):,}")

    # co_users distribution
    buckets = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    counts = [0] * (len(buckets) + 1)
    for cu, _ in co_occ.values():
        for i, threshold in enumerate(buckets):
            if cu <= threshold:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    total_pairs = len(co_occ)
    print("  co_users distribution:")
    prev = 0
    labels = [f"=={b}" if i == 0 else f"{buckets[i-1]+1}–{b}" for i, b in enumerate(buckets)]
    labels.append(f">{buckets[-1]}")
    for label, cnt in zip(labels, counts):
        if cnt:
            print(f"    co_users {label:>10s}: {cnt:>8,}  ({cnt/total_pairs*100:.1f}%)")

    return co_occ, item_user_cnt, qualified


# ---------------------------------------------------------------------------
# step 3: original Swing — Σ_{u,v ∈ U_AB} 1 / (α + |I_u ∩ I_v|)
#         normalized by sqrt(n_users_A × n_users_B) to suppress popularity bias
#
# For each pair (a, b), iterate over stored uid pairs and compute set
# intersection sizes.  u == v contributes once; u != v contributes × 2
# (both ordered pairs from the double sum).
# ---------------------------------------------------------------------------
def build_index(
    co_occ: dict,
    user_games: dict,
    qualified: set,
    item_user_cnt: dict,
    alpha: float,
    top_k: int,
) -> dict:
    # per-user qualified game sets (for fast intersection)
    user_game_sets: dict[int, set] = {
        uid: set(games.keys()) & qualified
        for uid, games in user_games.items()
    }

    neighbors: dict[int, list] = defaultdict(list)

    for (a, b), (co_u, uids) in co_occ.items():
        if co_u < MIN_CO_USERS:
            continue
        score = 0.0
        for i, u in enumerate(uids):
            gu = user_game_sets.get(u, set())
            for v in uids[i:]:          # includes self-pair (u == v)
                gv = user_game_sets.get(v, set())
                overlap = len(gu & gv)
                contrib = 1.0 / (alpha + overlap)
                score += contrib if u == v else 2 * contrib
        norm = math.sqrt(item_user_cnt.get(a, 1) * item_user_cnt.get(b, 1))
        score /= norm
        neighbors[a].append((score, b, co_u))
        neighbors[b].append((score, a, co_u))

    index = {}
    for gid, nbrs in neighbors.items():
        nbrs.sort(key=lambda x: -x[0])
        index[gid] = [
            {"game_id": int(b), "score": round(s, 6), "co_users": int(cu)}
            for s, b, cu in nbrs[:top_k]
            if cu >= MIN_CO_USERS
        ]
    return index


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".ndjson"))
    days = [f.split("_")[4] for f in data_files]  # extract YYYY-MM-DD
    print(f"Building i2i index from {len(days)} days: {days[0]} → {days[-1]}")

    t0 = time.time()

    print("\n[Pass 1] Collecting positive interactions per user...")
    user_games, n_records, n_pos = build_user_game_map(days)
    print(f"  unique users with >=1 positive game: {len(user_games):,}")
    print(f"  total records: {n_records:,}, positive interactions: {n_pos:,}")

    print("\n[Pass 2] Computing Swing co-occurrence...")
    co_occ, item_user_cnt, qualified = build_co_occurrence(user_games, MIN_ITEM_USERS)

    print("\n[Step 3] Building top-K neighbor index...")
    index = build_index(co_occ, user_games, qualified, item_user_cnt, ALPHA, TOP_K)
    print(f"  games with neighbors: {len(index):,}")

    # neighbor count distribution
    nbr_counts = [len(v) for v in index.values()]
    avg_nbr = sum(nbr_counts) / len(nbr_counts) if nbr_counts else 0
    print(f"  avg neighbors/game: {avg_nbr:.1f}, max: {max(nbr_counts) if nbr_counts else 0}")

    elapsed = time.time() - t0

    # write full index
    out_path = os.path.join(OUT_DIR, "i2i_index.json")
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(
            {str(k): v for k, v in index.items()},
            option=orjson.OPT_INDENT_2
        ))
    print(f"\nWrote {out_path}")

    # write simplified index: game_id -> [neighbor_game_id, ...]
    simple_path = os.path.join(OUT_DIR, "i2i_index_simple.json")
    with open(simple_path, "wb") as f:
        f.write(orjson.dumps(
            {str(k): [n["game_id"] for n in v] for k, v in index.items()},
            option=orjson.OPT_INDENT_2
        ))
    print(f"Wrote {simple_path}")

    # write stats
    stats = {
        "days": days,
        "n_records": n_records,
        "n_positive_interactions": n_pos,
        "n_users_with_history": len(user_games),
        "n_qualified_items": len(qualified),
        "n_co_occurrence_pairs": len(co_occ),
        "n_items_in_index": len(index),
        "avg_neighbors": round(avg_nbr, 2),
        "params": {
            "positive_gate": "label > 0",
            "MIN_ITEM_USERS": MIN_ITEM_USERS,
            "MIN_CO_USERS": MIN_CO_USERS,
            "TOP_K": TOP_K,
            "ALPHA": ALPHA,
            "MAX_PAIR_USERS": MAX_PAIR_USERS,
            "USER_GAME_CAP": USER_GAME_CAP,
        },
        "elapsed_sec": round(elapsed, 1),
    }
    stats_path = os.path.join(OUT_DIR, "i2i_stats.json")
    with open(stats_path, "wb") as f:
        f.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    print(f"Wrote {stats_path}")
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
