#!/usr/bin/env python3
"""
Feature pipeline for the two_tower_lite (粗排) two-tower model.

Input : GCS two_tower_lite/slim/{day}.ndjson.gz  (slim format, ~20 MB/day)
Output: artifacts/
          - config.json
          - vocab.json
          - train.npz
          - test.npz

Slim format (flat keys, see pull_and_build.py slim_record()):
  uid, country, platform, app_ver, game_id, author_id, category, tag, art,
  g_play/show/like/comment/remix/share/ver,
  a_fans/following/play/like/publish/comment/remix/share,
  create_ms, server_ms, play_time (label: None=neg, float=pos)
"""
import os, math, json, gzip, tempfile
import numpy as np
import orjson

OUT_DIR         = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
TRAIN_DAYS      = os.environ.get("TRAIN_DAYS", "2026-06-18,2026-06-19").split(",")
TEST_DAYS       = os.environ.get("TEST_DAYS",  "2026-06-20").split(",")
GCS_BUCKET      = os.environ.get("GCS_BUCKET", "")
GCS_SLIM_PREFIX = os.environ.get("GCS_SLIM_PREFIX", "two_tower_lite/slim")

PAD, OOV = 0, 1
MIN_FREQ = 1

# ── feature spec ──────────────────────────────────────────────────────────────
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

# numeric feature keys in slim format (order must match NUM_FEATURES)
NUM_SLIM_KEYS = [
    "g_play", "g_show", "g_like", "g_comment", "g_remix", "g_share", "g_ver",
    "a_fans", "a_following", "a_play", "a_like", "a_publish", "a_comment", "a_remix", "a_share",
]
NUM_FEATURES = [
    "game_play_cnt", "game_show_cnt", "game_like_cnt", "game_comment_cnt",
    "game_remix_cnt", "game_share_cnt", "game_version_cnt",
    "author_fans_cnt", "author_following_cnt", "author_play_cnt", "author_like_cnt",
    "author_publish_games", "author_comment_cnt", "author_remix_cnt", "author_share_cnt",
    "game_age_days",
]


# ── slim record accessors ─────────────────────────────────────────────────────

def cat_value(rec, feat):
    if feat == "user_id":       return rec.get("uid")
    if feat == "country_code":  return rec.get("country")
    if feat == "platform":      return rec.get("platform")
    if feat == "app_version":   return rec.get("app_ver")
    if feat == "game_id":       return rec.get("game_id")
    if feat == "author_id":     return rec.get("author_id")
    if feat == "game_category": return rec.get("category")
    if feat == "game_tag":      return rec.get("tag")
    if feat == "art_style":     return rec.get("art")
    raise KeyError(feat)


def norm_token(v):
    if v is None: return None
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return str(v)


def numeric_row(rec):
    vals = [rec.get(k) for k in NUM_SLIM_KEYS]
    cts = rec.get("create_ms"); sts = rec.get("server_ms")
    vals.append((sts - cts) / 86400000.0 if (cts and sts) else None)
    return vals


# ── download slim files from GCS ─────────────────────────────────────────────

def download_slim_days(days: list, local_dir: str) -> None:
    """Download slim gzip files for all days to local_dir."""
    if not GCS_BUCKET:
        return  # local dev: files already in DATA_DIR
    from google.cloud import storage as _gcs
    bucket = _gcs.Client().bucket(GCS_BUCKET)
    os.makedirs(local_dir, exist_ok=True)
    for day in days:
        blob_name  = f"{GCS_SLIM_PREFIX}/{day}.ndjson.gz"
        local_path = os.path.join(local_dir, f"{day}.ndjson.gz")
        if os.path.exists(local_path):
            continue
        bucket.blob(blob_name).download_to_filename(local_path)
        print(f"  ↓ {day}.ndjson.gz  ({os.path.getsize(local_path)/1e6:.1f}MB)", flush=True)


def iter_rows(days: list, local_dir: str):
    for day in days:
        path = os.path.join(local_dir, f"{day}.ndjson.gz")
        with gzip.open(path, "rb") as f:
            for line in f:
                if line.strip():
                    yield orjson.loads(line)


# ── pass 1: vocab + numeric stats ─────────────────────────────────────────────

def build_vocab(local_dir: str):
    from collections import defaultdict
    counts   = defaultdict(lambda: defaultdict(int))
    num_sum  = np.zeros(len(NUM_FEATURES))
    num_sqsum= np.zeros(len(NUM_FEATURES))
    num_cnt  = np.zeros(len(NUM_FEATURES))
    n_rows   = 0

    for rec in iter_rows(TRAIN_DAYS, local_dir):
        n_rows += 1
        for feat, _tower, vname in CAT_SINGLE:
            t = norm_token(cat_value(rec, feat))
            if t is not None:
                counts[vname][t] += 1
        for j, v in enumerate(numeric_row(rec)):
            if v is not None:
                lv = math.copysign(math.log1p(abs(v)), v)
                num_sum[j]   += lv
                num_sqsum[j] += lv * lv
                num_cnt[j]   += 1

    vocab = {}
    for name, c in counts.items():
        toks = [t for t, f in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))
                if f >= MIN_FREQ]
        vocab[name] = {tok: i + 2 for i, tok in enumerate(toks)}

    num_mean = np.where(num_cnt > 0, num_sum / np.maximum(num_cnt, 1), 0.0)
    num_var  = np.where(num_cnt > 0, num_sqsum / np.maximum(num_cnt, 1) - num_mean**2, 1.0)
    num_std  = np.sqrt(np.maximum(num_var, 1e-6))
    sizes    = {name: len(m) + 2 for name, m in vocab.items()}
    return vocab, sizes, num_mean, num_std, n_rows


# ── pass 2: encode split ──────────────────────────────────────────────────────

def encode_split(days: list, local_dir: str, vocab: dict, num_mean, num_std):
    N = sum(1 for _ in iter_rows(days, local_dir))
    D = len(NUM_FEATURES)

    out = {
        "label":        np.zeros(N, dtype=np.int8),
        "playing_time": np.zeros(N, dtype=np.float32),
        "uid_group":    np.zeros(N, dtype=np.int64),
        "num":          np.zeros((N, D), dtype=np.float32),
    }
    for feat, _t, _v in CAT_SINGLE:
        out[f"cat__{feat}"] = np.zeros(N, dtype=np.int32)

    oov_hits = {name: 0 for name in vocab}
    oov_tot  = {name: 0 for name in vocab}

    def enc(vname, raw):
        t = norm_token(raw)
        if t is None: return OOV
        return vocab[vname].get(t, OOV)

    for i, rec in enumerate(iter_rows(days, local_dir)):
        pt = rec.get("play_time")
        out["label"][i]        = 1 if pt is not None else 0
        out["playing_time"][i] = float(pt) if pt is not None else 0.0
        out["uid_group"][i]    = int(rec.get("uid") or 0)
        for feat, _t, vname in CAT_SINGLE:
            idx = enc(vname, cat_value(rec, feat))
            out[f"cat__{feat}"][i] = idx
            oov_tot[vname] += 1
            if idx == OOV: oov_hits[vname] += 1
        for jx, v in enumerate(numeric_row(rec)):
            if v is None:
                out["num"][i, jx] = 0.0
            else:
                lv = math.copysign(math.log1p(abs(v)), v)
                out["num"][i, jx] = (lv - num_mean[jx]) / num_std[jx]

    oov_rate = {k: round(oov_hits[k] / max(oov_tot[k], 1), 4) for k in vocab}
    return out, N, oov_rate


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_days = list(dict.fromkeys(TRAIN_DAYS + TEST_DAYS))  # dedup, preserve order

    with tempfile.TemporaryDirectory() as slim_dir:
        print(f"[download] fetching {len(all_days)} slim files...")
        download_slim_days(all_days, slim_dir)

        print(f"\n[pass 1] building vocab from train days: {TRAIN_DAYS}")
        vocab, sizes, num_mean, num_std, n_train = build_vocab(slim_dir)
        print(f"  train rows: {n_train:,}  MIN_FREQ={MIN_FREQ}")
        print("  vocab sizes (incl PAD+OOV):")
        for k, v in sizes.items():
            print(f"    {k:14s} {v}")

        with open(os.path.join(OUT_DIR, "vocab.json"), "wb") as f:
            f.write(orjson.dumps({k: {str(tk): iv for tk, iv in m.items()}
                                  for k, m in vocab.items()}))

        config = {
            "label":      {"name": "effective_play",
                           "rule": "play_time is not null",
                           "keep_raw": "playing_time"},
            "pad": PAD, "oov": OOV, "min_freq": MIN_FREQ,
            "cat_single": [{"feat": f, "tower": t, "vocab": v, "size": sizes[v]}
                           for f, t, v in CAT_SINGLE],
            "numeric":    {"features": NUM_FEATURES,
                           "transform": "signed_log1p_then_standardize",
                           "mean": num_mean.tolist(), "std": num_std.tolist()},
            "vocab_sizes": sizes,
            "excluded_as_leakage": ["position", "final_score", "predicted_scores",
                                    "pipeline", "reason_list", "context_info"],
        }
        with open(os.path.join(OUT_DIR, "config.json"), "wb") as f:
            f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))

        for split, days in [("train", TRAIN_DAYS), ("test", TEST_DAYS)]:
            print(f"\n[pass 2] encoding {split} ({len(days)} days)...")
            data, N, oov_rate = encode_split(days, slim_dir, vocab, num_mean, num_std)
            pos = float(data["label"].mean())
            print(f"  rows={N:,}  positive_rate={pos:.4f}")
            print(f"  OOV: " + "  ".join(f"{k}={v}" for k, v in oov_rate.items()))
            np.savez_compressed(os.path.join(OUT_DIR, f"{split}.npz"), **data)
            print(f"  wrote {split}.npz")

    print("\ndone.")


if __name__ == "__main__":
    main()
