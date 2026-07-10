#!/usr/bin/env python3
"""
Feature pipeline — slim mode (production).

Input : GCS two_tower_lite/slim/{YYYY-MM-DD}.ndjson.gz
        Each line is a flat JSON record produced by pull_and_build.py:slim_record().

Output: OUT_DIR/
          config.json   feature spec + vocab sizes + numeric normalization stats
          vocab.json    token → index maps  (0=PAD, 1=OOV)
          train.npz     encoded training rows
          test.npz      encoded test rows

Env vars:
  GCS_BUCKET          e.g. rezona-ml
  GCS_SLIM_PREFIX     e.g. two_tower_lite/slim
  TRAIN_DAYS          comma-separated dates, e.g. 2026-06-23,...
  TEST_DAYS           comma-separated dates, e.g. 2026-07-06
  OUT_DIR             local output directory
  MIN_FREQ            min token frequency to keep in vocab (default 1)
"""
import os, sys, math, json, gzip
import numpy as np
import orjson
from google.cloud import storage as gcs_lib

GCS_BUCKET      = os.environ["GCS_BUCKET"]
GCS_SLIM_PREFIX = os.environ["GCS_SLIM_PREFIX"]
TRAIN_DAYS      = os.environ["TRAIN_DAYS"].split(",")
TEST_DAYS       = os.environ["TEST_DAYS"].split(",")
OUT_DIR         = os.environ["OUT_DIR"]
MIN_FREQ        = int(os.environ.get("MIN_FREQ", 1))

PAD, OOV = 0, 1

# ── feature spec ──────────────────────────────────────────────────────────────
# (slim_key, feat_name, tower, vocab_name)
CAT_SINGLE = [
    ("uid",      "user_id",       "user", "user_id"),
    ("country",  "country_code",  "user", "country"),
    ("platform", "platform",      "user", "platform"),
    ("app_ver",  "app_version",   "user", "app_version"),
    ("game_id",  "game_id",       "item", "game_id"),
    ("author_id","author_id",     "item", "author_id"),
    ("category", "game_category", "item", "category"),
    ("tag",      "game_tag",      "item", "tag"),
    ("art",      "art_style",     "item", "art_style"),
]

# slim key order → must match model's numeric input (no game_age_days)
NUM_SLIM_KEYS = [
    "g_play", "g_show", "g_like", "g_comment", "g_remix", "g_share", "g_ver",
    "a_fans", "a_following", "a_play", "a_like", "a_publish", "a_comment", "a_remix", "a_share",
]
NUM_FEATURES = [
    "game_play_cnt", "game_show_cnt", "game_like_cnt", "game_comment_cnt",
    "game_remix_cnt", "game_share_cnt", "game_version_cnt",
    "author_fans_cnt", "author_following_cnt", "author_play_cnt", "author_like_cnt",
    "author_publish_games", "author_comment_cnt", "author_remix_cnt", "author_share_cnt",
]


# ── GCS download ──────────────────────────────────────────────────────────────

def download_slim_days(days: list, local_dir: str) -> None:
    bucket = gcs_lib.Client().bucket(GCS_BUCKET)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download] fetching {len(days)} slim files...")
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


# ── helpers ───────────────────────────────────────────────────────────────────

def norm_token(v):
    if v is None: return None
    s = str(v).strip()
    return s if s else None


def numeric_row(rec):
    return [rec.get(k) for k in NUM_SLIM_KEYS]


# ── pass 1: vocab + numeric stats (train only) ────────────────────────────────

def build_vocab(local_dir: str):
    from collections import defaultdict
    counts   = defaultdict(lambda: defaultdict(int))
    num_sum  = np.zeros(len(NUM_FEATURES))
    num_sqsum= np.zeros(len(NUM_FEATURES))
    num_cnt  = np.zeros(len(NUM_FEATURES))
    n_rows   = 0

    for rec in iter_rows(TRAIN_DAYS, local_dir):
        n_rows += 1
        for slim_key, _feat, _tower, vocab in CAT_SINGLE:
            t = norm_token(rec.get(slim_key))
            if t is not None:
                counts[vocab][t] += 1

        for j, v in enumerate(numeric_row(rec)):
            if v is not None:
                lv = math.copysign(math.log1p(abs(v)), v)
                num_sum[j]   += lv
                num_sqsum[j] += lv * lv
                num_cnt[j]   += 1

    print(f"  train rows: {n_rows:,}  MIN_FREQ={MIN_FREQ}")

    vocab = {}
    for name, c in counts.items():
        toks = [t for t, f in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))) if f >= MIN_FREQ]
        vocab[name] = {tok: i + 2 for i, tok in enumerate(toks)}

    num_mean = np.where(num_cnt > 0, num_sum / np.maximum(num_cnt, 1), 0.0)
    num_var  = np.where(num_cnt > 0, num_sqsum / np.maximum(num_cnt, 1) - num_mean**2, 1.0)
    num_std  = np.sqrt(np.maximum(num_var, 1e-6))

    sizes = {name: len(m) + 2 for name, m in vocab.items()}
    print("  vocab sizes (incl PAD+OOV):")
    for k, v in sizes.items():
        print(f"    {k:14s} {v}")
    return vocab, sizes, num_mean, num_std


# ── pass 2: encode split ───────────────────────────────────────────────────────

def encode_split(days, local_dir, vocab, num_mean, num_std, split_name: str):
    N = sum(1 for _ in iter_rows(days, local_dir))
    D = len(NUM_FEATURES)

    out = {
        "label":     np.zeros(N, dtype=np.int8),
        "uid_group": np.zeros(N, dtype=np.int64),
        "num":       np.zeros((N, D), dtype=np.float32),
    }
    for _slim_key, feat, _tower, _vocab in CAT_SINGLE:
        out[f"cat__{feat}"] = np.zeros(N, dtype=np.int32)

    oov_hits = {name: 0 for name in vocab}
    oov_tot  = {name: 0 for name in vocab}

    for i, rec in enumerate(iter_rows(days, local_dir)):
        out["label"][i]     = 1 if rec.get("play_time") is not None else 0
        uid_raw = rec.get("uid")
        out["uid_group"][i] = int(uid_raw) if uid_raw is not None else 0

        for slim_key, feat, _tower, vname in CAT_SINGLE:
            t   = norm_token(rec.get(slim_key))
            idx = vocab[vname].get(t, OOV) if t is not None else OOV
            out[f"cat__{feat}"][i] = idx
            oov_tot[vname]  += 1
            if idx == OOV: oov_hits[vname] += 1

        for j, v in enumerate(numeric_row(rec)):
            if v is not None:
                lv = math.copysign(math.log1p(abs(v)), v)
                out["num"][i, j] = (lv - num_mean[j]) / num_std[j]

    pos = float(out["label"].mean())
    oov_rate = {k: round(oov_hits[k] / max(oov_tot[k], 1), 4) for k in vocab}
    print(f"  rows={N:,}  positive_rate={pos:.4f}")
    print(f"  OOV: " + "  ".join(f"{k}={v}" for k, v in oov_rate.items()))
    np.savez_compressed(os.path.join(OUT_DIR, f"{split_name}.npz"), **out)
    print(f"  wrote {split_name}.npz")
    return N, oov_rate


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as slim_dir:
        download_slim_days(TRAIN_DAYS + TEST_DAYS, slim_dir)

        print(f"\n[pass 1] building vocab from train days: {TRAIN_DAYS}")
        vocab, sizes, num_mean, num_std = build_vocab(slim_dir)

        config = {
            "cat_single": [{"feat": feat, "tower": tower, "vocab": vname, "size": sizes[vname]}
                           for _slim_key, feat, tower, vname in CAT_SINGLE],
            "numeric":    {"features": NUM_FEATURES, "transform": "signed_log1p_then_standardize",
                           "mean": num_mean.tolist(), "std": num_std.tolist()},
            "vocab_sizes": sizes,
            "pad": PAD, "oov": OOV,
        }
        with open(os.path.join(OUT_DIR, "config.json"), "wb") as f:
            f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))
        with open(os.path.join(OUT_DIR, "vocab.json"), "wb") as f:
            f.write(orjson.dumps({k: {str(tk): iv for tk, iv in m.items()} for k, m in vocab.items()}))

        for split_name, days in [("train", TRAIN_DAYS), ("test", TEST_DAYS)]:
            print(f"\n[pass 2] encoding {split_name} ({len(days)} days)...")
            encode_split(days, slim_dir, vocab, num_mean, num_std, split_name)

    print("\ndone.")


if __name__ == "__main__":
    main()
