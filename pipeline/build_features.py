#!/usr/bin/env python3
"""
Feature pipeline for the coarse-ranking (粗排) two-tower model.

Input : data/reco_labeled_log_flat_YYYY-MM-DD_realshow.ndjson  (one impression per line)
Output: artifacts/
          - config.json        feature spec + vocab sizes + numeric normalization stats
          - vocab.json         id -> index maps (reserved: 0=PAD, 1=OOV)
          - train.npz          features for 2026-06-18 + 2026-06-19
          - test.npz           features for 2026-06-20

Label : effective play. positive = playing_time is present (user actually opened/played).
        We also keep raw playing_time so the threshold can be changed later without re-parsing.

Design notes
------------
* Two-tower split — features are tagged user / item / context so the model code can wire
  them into the right tower. position / final_score / predicted_scores / pipeline / reason_list
  are SERVING OUTPUTS and are deliberately NOT used as features (leakage + position bias).
* Vocabularies are built from the TRAIN split only; test rows that miss the vocab map to OOV(1).
* Shared vocabularies: game_id (target + behavior seq), game_tag (target + seq + user_top_tags),
  game_category (target + user_top_categories) — so history and target live in one embedding space.
"""
import os, sys, json, math
import numpy as np
import orjson

DATA_DIR   = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
OUT_DIR    = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
TRAIN_DAYS = os.environ.get("TRAIN_DAYS", "2026-06-18,2026-06-19").split(",")
TEST_DAYS  = os.environ.get("TEST_DAYS", "2026-06-20").split(",")
FILE_TMPL  = "reco_labeled_log_flat_{day}_realshow.ndjson"

K_SEQ      = 50   # truncate behavior sequence to most-recent K
MAX_MH     = 10   # truncate user_top_categories / user_top_tags
PAD, OOV   = 0, 1
MIN_FREQ   = 1    # min train frequency to keep a token in vocab

# ---- feature spec ---------------------------------------------------------
# single categorical: (feature_name, tower, vocab_name)  vocab_name allows sharing
CAT_SINGLE = [
    ("user_id",       "user", "user_id"),
    ("country_code",  "user", "country"),
    ("platform",      "user", "platform"),
    ("app_version",   "user", "app_version"),
    ("game_id",       "item", "game_id"),
    ("author_id",     "item", "author_id"),
    ("game_category", "item", "category"),
    ("game_tag",      "item", "tag"),
    ("art_style",     "item", "art_style"),
]
# behavior sequence (user tower): (out_name, vocab_name)  -- numeric play_time handled separately
SEQ_CAT = [
    ("seq_game_id",  "game_id"),
    ("seq_game_tag", "tag"),
    ("seq_inter",    "interaction"),
]
# multi-hot (user tower): (out_name, source_field, vocab_name)
MULTIHOT = [
    ("mh_topcat", "user_top_categories", "category"),
    ("mh_toptag", "user_top_tags",       "tag"),
]
# numeric (item tower): (out_name, path)  -- path is dotted into game_info
NUM_GAME_STATS   = ["play_cnt","show_cnt","like_cnt","comment_cnt","remix_cnt","share_cnt","game_version_cnt"]
NUM_AUTHOR_STATS = ["fans_cnt","following_cnt","play_cnt","like_cnt","publish_games","comment_cnt","remix_cnt","share_cnt"]
NUM_FEATURES = (
    [f"game_{k}"   for k in NUM_GAME_STATS] +
    [f"author_{k}" for k in NUM_AUTHOR_STATS] +
    ["game_age_days"]
)


def iter_rows(days):
    for day in days:
        path = os.path.join(DATA_DIR, FILE_TMPL.format(day=day))
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    yield orjson.loads(line)


# ---------------------------------------------------------------------------
# helpers to pull raw token values / numeric values out of a record
# ---------------------------------------------------------------------------
def cat_value(rec, feat):
    u, g = rec["user_info"], rec["game_info"]
    if feat == "user_id":       return u.get("user_id")
    if feat == "country_code":  return rec.get("country_code")
    if feat == "platform":      return rec.get("platform")
    if feat == "app_version":   return rec.get("app_version")
    if feat == "game_id":       return g.get("game_id")
    if feat == "author_id":     return g.get("author_id")
    if feat == "game_category": return g.get("game_category")
    if feat == "game_tag":      return g.get("game_tag")
    if feat == "art_style":     return g.get("art_style")
    raise KeyError(feat)


def norm_token(v):
    """Normalize a raw token to a stable string key; None/'' -> None (will map to OOV)."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return str(v)


def recent_games(rec):
    return rec["user_info"].get("recent_games") or []


def numeric_row(rec):
    g = rec["game_info"]
    gs = g.get("game_stats") or {}
    as_ = g.get("author_stats") or {}
    out = []
    for k in NUM_GAME_STATS:
        out.append(gs.get(k))
    for k in NUM_AUTHOR_STATS:
        out.append(as_.get(k))
    # game age in days at request time
    cts = g.get("create_ts_ms"); sts = rec.get("server_ts_ms")
    age = (sts - cts) / 86400000.0 if (cts and sts) else None
    out.append(age)
    return out  # list len == len(NUM_FEATURES), entries may be None


# ---------------------------------------------------------------------------
# pass 1: build vocabularies + numeric stats from TRAIN only
# ---------------------------------------------------------------------------
def build_vocab():
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))   # vocab_name -> token -> count
    # numeric: collect for log1p mean/std (signed-log1p for age which can be tiny/neg)
    num_sum  = np.zeros(len(NUM_FEATURES))
    num_sqsum= np.zeros(len(NUM_FEATURES))
    num_cnt  = np.zeros(len(NUM_FEATURES))
    n_rows = 0

    for rec in iter_rows(TRAIN_DAYS):
        n_rows += 1
        # single categoricals
        for feat, _tower, vocab in CAT_SINGLE:
            t = norm_token(cat_value(rec, feat))
            if t is not None:
                counts[vocab][t] += 1
        # behavior sequence (last K)
        for r in recent_games(rec)[-K_SEQ:]:
            gt = norm_token(r.get("game_id"))
            if gt is not None: counts["game_id"][gt] += 1
            tg = norm_token(r.get("game_tag"))
            if tg is not None: counts["tag"][tg] += 1
            il = norm_token(r.get("interaction_label"))
            if il is not None: counts["interaction"][il] += 1
        # multi-hot
        for _out, field, vocab in MULTIHOT:
            for v in (rec["user_info"].get(field) or [])[:MAX_MH]:
                t = norm_token(v)
                if t is not None: counts[vocab][t] += 1
        # numeric -> accumulate log1p stats
        vals = numeric_row(rec)
        for j, v in enumerate(vals):
            if v is not None:
                lv = math.copysign(math.log1p(abs(v)), v)
                num_sum[j]   += lv
                num_sqsum[j] += lv * lv
                num_cnt[j]   += 1

    # finalize vocabs: index 0=PAD, 1=OOV, then tokens sorted by freq desc
    vocab = {}
    for name, c in counts.items():
        toks = [t for t, f in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))) if f >= MIN_FREQ]
        m = {PAD: PAD, OOV: OOV}  # placeholder; real map below
        mapping = {tok: i + 2 for i, tok in enumerate(toks)}
        vocab[name] = mapping

    num_mean = np.where(num_cnt > 0, num_sum / np.maximum(num_cnt, 1), 0.0)
    num_var  = np.where(num_cnt > 0, num_sqsum / np.maximum(num_cnt, 1) - num_mean**2, 1.0)
    num_std  = np.sqrt(np.maximum(num_var, 1e-6))

    sizes = {name: len(m) + 2 for name, m in vocab.items()}  # +2 for PAD/OOV
    return vocab, sizes, num_mean, num_std, n_rows


# ---------------------------------------------------------------------------
# pass 2: encode a split to fixed-shape numpy arrays
# ---------------------------------------------------------------------------
def encode_split(days, vocab, num_mean, num_std):
    # stream: count first (cheap), allocate, then fill row-by-row without holding all dicts
    N = sum(1 for _ in iter_rows(days))
    D = len(NUM_FEATURES)

    out = {
        "label":        np.zeros(N, dtype=np.int8),
        "playing_time": np.zeros(N, dtype=np.float32),
        "uid_group":    np.zeros(N, dtype=np.int64),   # raw user_id for GAUC grouping
        "num":          np.zeros((N, D), dtype=np.float32),
        "seq_len":      np.zeros(N, dtype=np.int32),
    }
    for feat, _t, _v in CAT_SINGLE:
        out[f"cat__{feat}"] = np.zeros(N, dtype=np.int32)
    for name, _v in SEQ_CAT:
        out[f"seq__{name}"] = np.zeros((N, K_SEQ), dtype=np.int32)
    out["seq__play_time"] = np.zeros((N, K_SEQ), dtype=np.float32)
    for name, _f, _v in MULTIHOT:
        out[f"mh__{name}"]    = np.zeros((N, MAX_MH), dtype=np.int32)
        out[f"mhlen__{name}"] = np.zeros(N, dtype=np.int32)

    # play_time normalization (reuse log1p standardize with its own stats from this seq? use train num? )
    # behavior play_time is in seconds; standardize via log1p with simple fixed stats computed here lazily.
    def enc(vocab_name, raw):
        t = norm_token(raw)
        if t is None: return OOV
        return vocab[vocab_name].get(t, OOV)

    oov_hits = {name: 0 for name in vocab}
    oov_tot  = {name: 0 for name in vocab}

    for i, rec in enumerate(iter_rows(days)):
        pt = rec.get("context_info", {}).get("playing_time")
        out["label"][i]        = 1 if pt is not None else 0
        out["playing_time"][i] = float(pt) if pt is not None else 0.0
        out["uid_group"][i]    = int(rec["user_info"].get("user_id") or 0)

        for feat, _t, vname in CAT_SINGLE:
            idx = enc(vname, cat_value(rec, feat))
            out[f"cat__{feat}"][i] = idx
            oov_tot[vname] += 1
            if idx == OOV: oov_hits[vname] += 1

        # behavior sequence: take most-recent K, left-pad with 0
        rg = recent_games(rec)[-K_SEQ:]
        L = len(rg)
        out["seq_len"][i] = L
        for j, r in enumerate(rg):
            pos = K_SEQ - L + j  # right-align (most recent at the end)
            out["seq__seq_game_id"][i, pos]  = enc("game_id", r.get("game_id"))
            out["seq__seq_game_tag"][i, pos] = enc("tag", r.get("game_tag"))
            out["seq__seq_inter"][i, pos]    = enc("interaction", r.get("interaction_label"))
            ptv = r.get("play_time")
            out["seq__play_time"][i, pos] = math.log1p(ptv) if (ptv and ptv > 0) else 0.0

        for name, field, vname in MULTIHOT:
            vals = (rec["user_info"].get(field) or [])[:MAX_MH]
            for j, v in enumerate(vals):
                out[f"mh__{name}"][i, j] = enc(vname, v)
            out[f"mhlen__{name}"][i] = len(vals)

        vals = numeric_row(rec)
        for jx, v in enumerate(vals):
            if v is None:
                out["num"][i, jx] = 0.0  # standardized missing -> mean
            else:
                lv = math.copysign(math.log1p(abs(v)), v)
                out["num"][i, jx] = (lv - num_mean[jx]) / num_std[jx]

    oov_rate = {k: round(oov_hits[k] / max(oov_tot[k], 1), 4) for k in vocab}
    return out, N, oov_rate


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[pass 1] building vocab + numeric stats from train days:", TRAIN_DAYS)
    vocab, sizes, num_mean, num_std, n_train = build_vocab()
    print(f"  train rows: {n_train}")
    print("  vocab sizes (incl PAD+OOV):")
    for k, v in sizes.items():
        print(f"    {k:14s} {v}")

    # persist vocab + config
    with open(os.path.join(OUT_DIR, "vocab.json"), "wb") as f:
        f.write(orjson.dumps({k: {str(tk): iv for tk, iv in m.items()} for k, m in vocab.items()}))

    config = {
        "label": {"name": "effective_play", "rule": "playing_time is not null", "keep_raw": "playing_time"},
        "k_seq": K_SEQ, "max_multihot": MAX_MH, "pad": PAD, "oov": OOV,
        "cat_single": [{"feat": f, "tower": t, "vocab": v, "size": sizes[v]} for f, t, v in CAT_SINGLE],
        "seq_cat":    [{"name": n, "vocab": v, "size": sizes[v]} for n, v in SEQ_CAT],
        "seq_numeric": ["play_time"],
        "multihot":   [{"name": n, "field": fld, "vocab": v, "size": sizes[v]} for n, fld, v in MULTIHOT],
        "numeric":    {"features": NUM_FEATURES, "transform": "signed_log1p_then_standardize",
                        "mean": num_mean.tolist(), "std": num_std.tolist()},
        "vocab_sizes": sizes,
        "excluded_as_leakage": ["position","final_score","predicted_scores","pipeline","reason_list","context_info"],
    }
    with open(os.path.join(OUT_DIR, "config.json"), "wb") as f:
        f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))

    for split, days in [("train", TRAIN_DAYS), ("test", TEST_DAYS)]:
        print(f"[pass 2] encoding {split}: {days}")
        data, N, oov_rate = encode_split(days, vocab, num_mean, num_std)
        pos = float(data["label"].mean())
        print(f"  rows={N}  positive_rate={pos:.4f}")
        print(f"  test OOV rates: " + ", ".join(f"{k}={v}" for k, v in oov_rate.items()))
        np.savez_compressed(os.path.join(OUT_DIR, f"{split}.npz"), **data)
        print(f"  wrote artifacts/{split}.npz")

    print("done.")


if __name__ == "__main__":
    main()
