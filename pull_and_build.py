#!/usr/bin/env python3
"""
Job 1 — 增量拉取 BQ + slim 化 + 上传 GCS + 构建特征。

数据流:
  BQ → slim ndjson.gz (只保留特征字段，打平结构) → GCS slim/{day}.ndjson.gz
  build_features.py 下载 slim 文件 → 建 vocab → 编码 → train/test npz

Slim 格式比原始 ndjson 小约 50x（~20 MB/天 vs ~1.5 GB/天），
build 阶段下载所有天约几百 MB，无需流式读大文件。

Env vars:
  GCS_BUCKET              e.g. rezona-ml
  GCS_ARTIFACTS_PREFIX    e.g. two_tower_lite/artifacts
  GCS_SLIM_PREFIX         e.g. two_tower_lite/slim   (default)
  GCS_RAW_PREFIX          e.g. two_tower_lite/raw    (migration: slim from existing raw)
  BQ_PROJECT              default: rezonaai
  BQ_TABLE                default: rezonaai.datalake.reco_labeled_log_flat
  BQ_WHERE                default: context_info.realshow = TRUE
  BQ_LIMIT                max rows per day, 0=unlimited
  BQ_TIMEOUT_SEC          default: 1800
  TRAIN_DAYS / TEST_DAYS  override auto window
"""
import os, sys, subprocess, datetime, tempfile, time, decimal, gzip
from pathlib import Path

from google.cloud import bigquery
from google.cloud import storage as gcs
import orjson

REPO_ROOT = Path(__file__).resolve().parent

# ── config ────────────────────────────────────────────────────────────────────
BQ_PROJECT   = os.environ.get("BQ_PROJECT", "rezonaai")
BQ_TABLE     = os.environ.get("BQ_TABLE", "rezonaai.datalake.reco_labeled_log_flat")
BQ_PART_COL  = os.environ.get("BQ_PARTITION_COL", "dt")
BQ_WHERE     = os.environ.get("BQ_WHERE", "context_info.realshow = TRUE").strip()
BQ_LIMIT     = int(os.environ.get("BQ_LIMIT", 0))
BQ_TIMEOUT   = int(os.environ.get("BQ_TIMEOUT_SEC", 1800))
PROGRESS_N   = 10_000

GCS_BUCKET      = os.environ["GCS_BUCKET"]
GCS_ART_PREFIX  = os.environ["GCS_ARTIFACTS_PREFIX"]
GCS_SLIM_PREFIX = os.environ.get("GCS_SLIM_PREFIX", "two_tower_lite/slim")
GCS_RAW_PREFIX  = os.environ.get("GCS_RAW_PREFIX",  "two_tower_lite/raw")  # migration only


# ── slim: extract only fields needed by build_features.py ────────────────────

def slim_record(rec: dict) -> dict:
    """Flatten and filter a full BQ row to only the fields used for training."""
    u   = rec.get("user_info")  or {}
    g   = rec.get("game_info")  or {}
    gs  = g.get("game_stats")   or {}
    as_ = g.get("author_stats") or {}
    pt  = (rec.get("context_info") or {}).get("playing_time")
    return {
        "uid":       u.get("user_id"),
        "country":   rec.get("country_code"),
        "platform":  rec.get("platform"),
        "app_ver":   rec.get("app_version"),
        "game_id":   g.get("game_id"),
        "author_id": g.get("author_id"),
        "category":  g.get("game_category"),
        "tag":       g.get("game_tag"),
        "art":       g.get("art_style"),
        # game stats (7)
        "g_play":    gs.get("play_cnt"),
        "g_show":    gs.get("show_cnt"),
        "g_like":    gs.get("like_cnt"),
        "g_comment": gs.get("comment_cnt"),
        "g_remix":   gs.get("remix_cnt"),
        "g_share":   gs.get("share_cnt"),
        "g_ver":     gs.get("game_version_cnt"),
        # author stats (8)
        "a_fans":      as_.get("fans_cnt"),
        "a_following": as_.get("following_cnt"),
        "a_play":      as_.get("play_cnt"),
        "a_like":      as_.get("like_cnt"),
        "a_publish":   as_.get("publish_games"),
        "a_comment":   as_.get("comment_cnt"),
        "a_remix":     as_.get("remix_cnt"),
        "a_share":     as_.get("share_cnt"),
        # timestamps
        "create_ms":   g.get("create_ts_ms"),
        "server_ms":   rec.get("server_ts_ms"),
        # label
        "play_time":   pt,
    }


# ── normalize: BQ type → JSON-serializable ────────────────────────────────────

def normalize(v):
    if isinstance(v, dict):   return {k: normalize(vv) for k, vv in v.items()}
    if isinstance(v, list):   return [normalize(i) for i in v]
    if isinstance(v, datetime.datetime): return v.isoformat()
    if isinstance(v, datetime.date):     return v.isoformat()
    if isinstance(v, datetime.time):     return v.isoformat()
    if isinstance(v, decimal.Decimal):   return float(v)
    if isinstance(v, bytes):
        import base64; return base64.b64encode(v).decode()
    return v


# ── BQ helpers ────────────────────────────────────────────────────────────────

def bq_client():
    return bigquery.Client(project=BQ_PROJECT)


def count_day(bq: bigquery.Client, day: str) -> int:
    extra = f" AND ({BQ_WHERE})" if BQ_WHERE else ""
    sql   = (f"SELECT COUNT(*) AS n FROM `{BQ_TABLE}`"
             f" WHERE {BQ_PART_COL} = @day" + extra)
    job   = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("day", "DATE", day)]
    ))
    return list(job.result())[0]["n"]


def pull_day_slim(bq: bigquery.Client, day: str, local_gz: str) -> int:
    """Pull one day from BQ, slim on the fly, write gzip ndjson. Returns row count."""
    extra        = f" AND ({BQ_WHERE})" if BQ_WHERE else ""
    limit_clause = f" LIMIT {BQ_LIMIT}" if BQ_LIMIT > 0 else ""
    sql = (f"SELECT * FROM `{BQ_TABLE}`"
           f" WHERE {BQ_PART_COL} = @day" + extra + limit_clause)

    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("day", "DATE", day)],
        job_timeout_ms=BQ_TIMEOUT * 1000,
    ))

    written = 0
    t0 = time.time()
    with gzip.open(local_gz, "wb") as f:
        for row in job.result():
            f.write(orjson.dumps(slim_record(normalize(dict(row)))) + b"\n")
            written += 1
            if written % PROGRESS_N == 0:
                print(f"  ... {written:,} rows ({time.time()-t0:.0f}s)", flush=True)

    print(f"✓ {day}: {written:,} rows  ({time.time()-t0:.0f}s)  "
          f"slim={os.path.getsize(local_gz)/1e6:.1f}MB")
    return written


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _bucket():
    return gcs.Client().bucket(GCS_BUCKET)

def gcs_exists(bucket, blob_name: str) -> bool:
    return bucket.blob(blob_name).exists()

def gcs_upload(local_path: str, bucket, blob_name: str) -> None:
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"  → gs://{GCS_BUCKET}/{blob_name}  ({os.path.getsize(local_path)/1e6:.1f}MB)")

def gcs_download(bucket, blob_name: str, local_path: str) -> None:
    bucket.blob(blob_name).download_to_filename(local_path)


# ── step 1: ensure slim files in GCS ─────────────────────────────────────────

def ensure_slim_in_gcs(all_days: list) -> None:
    """For each day: check slim cache → fall back to raw cache → pull from BQ."""
    bucket = _bucket()

    missing = [d for d in all_days
               if not gcs_exists(bucket, f"{GCS_SLIM_PREFIX}/{d}.ndjson.gz")]

    if len(all_days) - len(missing):
        print(f"[slim cache] hit {len(all_days)-len(missing)} days")
    if not missing:
        print("[slim] all days cached"); return

    bq = None
    with tempfile.TemporaryDirectory() as tmp:
        for day in missing:
            slim_blob = f"{GCS_SLIM_PREFIX}/{day}.ndjson.gz"
            local_gz  = os.path.join(tmp, f"{day}.ndjson.gz")
            raw_blob  = f"{GCS_RAW_PREFIX}/{day}.ndjson"

            if gcs_exists(bucket, raw_blob):
                # migration: slim existing raw file
                print(f"[slim] {day}: converting from raw cache...")
                local_raw = os.path.join(tmp, f"{day}.ndjson")
                gcs_download(bucket, raw_blob, local_raw)
                written = 0
                with open(local_raw, "rb") as fin, gzip.open(local_gz, "wb") as fout:
                    for line in fin:
                        if line.strip():
                            rec = orjson.loads(line)
                            fout.write(orjson.dumps(slim_record(rec)) + b"\n")
                            written += 1
                os.unlink(local_raw)
                print(f"  slimmed {written:,} rows  "
                      f"slim={os.path.getsize(local_gz)/1e6:.1f}MB")
            else:
                # pull from BQ
                if bq is None:
                    bq = bq_client()
                    print(f"[BQ] {BQ_TABLE}  filter: {BQ_WHERE or '(none)'}")
                n = count_day(bq, day)
                print(f"[BQ] {day}: {n:,} rows")
                if n == 0:
                    print(f"  skip"); continue
                pull_day_slim(bq, day, local_gz)

            gcs_upload(local_gz, bucket, slim_blob)
            os.unlink(local_gz)


# ── step 2: build features ────────────────────────────────────────────────────

def build_features(art_dir: str, train_days: list, test_days: list) -> None:
    print(f"\n[features] building artifacts -> {art_dir}")
    env = {
        **os.environ,
        "TRAIN_DAYS":      ",".join(train_days),
        "TEST_DAYS":       ",".join(test_days),
        "OUT_DIR":         art_dir,
        "GCS_BUCKET":      GCS_BUCKET,
        "GCS_SLIM_PREFIX": GCS_SLIM_PREFIX,
    }
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "pipeline" / "build_features.py")],
        env=env, check=True,
    )


# ── step 3: upload artifacts ──────────────────────────────────────────────────

def upload_artifacts(art_dir: str) -> None:
    prefix = f"{GCS_ART_PREFIX}/latest"
    bucket = _bucket()
    print(f"\n[GCS] uploading artifacts -> gs://{GCS_BUCKET}/{prefix}/")
    for fname in ["train.npz", "test.npz", "config.json", "vocab.json"]:
        local = os.path.join(art_dir, fname)
        if os.path.exists(local):
            gcs_upload(local, bucket, f"{prefix}/{fname}")
        else:
            print(f"  WARNING: {fname} not found, skipping")


# ── resolve days ──────────────────────────────────────────────────────────────

def resolve_days():
    train_env = os.environ.get("TRAIN_DAYS", "").strip()
    test_env  = os.environ.get("TEST_DAYS",  "").strip()
    if train_env and test_env:
        return train_env.split(","), test_env.split(",")

    today    = datetime.date.today()
    test_dt  = today - datetime.timedelta(days=1)
    start_dt = test_dt - datetime.timedelta(days=13)
    train_days = [(start_dt + datetime.timedelta(days=i)).isoformat() for i in range(13)]
    test_days  = [test_dt.isoformat()]
    print(f"today={today}  train=[{train_days[0]}, {train_days[-1]}]  test={test_days}")
    return train_days, test_days


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    train_days, test_days = resolve_days()

    ensure_slim_in_gcs(train_days + test_days)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = os.path.join(tmpdir, "artifacts")
        os.makedirs(art_dir)
        build_features(art_dir, train_days, test_days)
        upload_artifacts(art_dir)

    print("\ndone.")


if __name__ == "__main__":
    main()
