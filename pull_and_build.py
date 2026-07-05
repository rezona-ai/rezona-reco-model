#!/usr/bin/env python3
"""
Job 1 — 增量拉取 BQ + 特征构建 + 上传 GCS。

流程:
  检查 GCS raw/ 里缺哪些天
    → 只从 BQ 拉缺失的天（通常只有今天 1 天）
    → 写入 GCS raw/<day>.ndjson
    → build_features.py 从本地临时目录读取
    → 上传 artifacts 到 GCS_ARTIFACTS_PREFIX/latest/

Env vars:
  GCS_BUCKET              e.g. rezona-ml
  GCS_ARTIFACTS_PREFIX    e.g. coarse-ranking/artifacts
  GCS_RAW_PREFIX          e.g. coarse-ranking/raw  (per-day ndjson 缓存)
  TRAIN_DAYS              逗号分隔；留空则自动取最近 13 天
  TEST_DAYS               逗号分隔；留空则自动取昨天
  BQ_PROJECT              default: rezonaai
  BQ_TABLE                default: rezonaai.datalake.reco_labeled_log_flat
  BQ_PARTITION_COL        default: dt
  BQ_LIMIT                max rows per day (0=unlimited, debug用)
"""
import os, sys, subprocess, datetime, tempfile
from pathlib import Path

from google.cloud import bigquery
from google.cloud import storage as gcs
import orjson

REPO_ROOT = Path(__file__).resolve().parent

# ── config ────────────────────────────────────────────────────────────────────
BQ_PROJECT  = os.environ.get("BQ_PROJECT", "rezonaai")
BQ_TABLE    = os.environ.get("BQ_TABLE", "rezonaai.datalake.reco_labeled_log_flat")
BQ_PART_COL = os.environ.get("BQ_PARTITION_COL", "dt")
BQ_LIMIT    = int(os.environ.get("BQ_LIMIT", 0))

GCS_BUCKET     = os.environ["GCS_BUCKET"]
GCS_ART_PREFIX = os.environ["GCS_ARTIFACTS_PREFIX"]          # e.g. coarse-ranking/artifacts
GCS_RAW_PREFIX = os.environ.get("GCS_RAW_PREFIX", "coarse-ranking/raw")

today     = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

def _default_train_days(n=13):
    start = yesterday - datetime.timedelta(days=n - 1)
    return [(start + datetime.timedelta(days=i)).isoformat() for i in range(n)]

TRAIN_DAYS = os.environ.get("TRAIN_DAYS", ",".join(_default_train_days())).split(",")
TEST_DAYS  = os.environ.get("TEST_DAYS", yesterday.isoformat()).split(",")

FILE_TMPL = "reco_labeled_log_flat_{day}_realshow.ndjson"   # matches build_features.py


# ── GCS helpers ───────────────────────────────────────────────────────────────

def gcs_exists(bucket, blob_name: str) -> bool:
    return bucket.blob(blob_name).exists()

def gcs_upload(local_path: str, bucket, blob_name: str) -> None:
    bucket.blob(blob_name).upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  uploaded {os.path.basename(local_path)} ({size_mb:.1f}MB) "
          f"-> gs://{GCS_BUCKET}/{blob_name}")

def gcs_download(bucket, blob_name: str, local_path: str) -> None:
    bucket.blob(blob_name).download_to_filename(local_path)


# ── step 1: incremental BQ pull → GCS raw/ ───────────────────────────────────

def pull_day(bq: bigquery.Client, day: str, local_path: str) -> int:
    import time as _time
    limit_clause = f" LIMIT {BQ_LIMIT}" if BQ_LIMIT > 0 else ""
    sql = (
        f"SELECT * FROM `{BQ_TABLE}`"
        f" WHERE {BQ_PART_COL} = @day" + limit_clause
    )
    job = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("day", "DATE", day)]
        ),
    )
    written = 0
    t0 = _time.time()
    with open(local_path, "wb") as f:
        for row in job.result():
            f.write(orjson.dumps(dict(row)) + b"\n")
            written += 1
            if written % 50_000 == 0:
                print(f"    {day}: {written} rows ({_time.time()-t0:.0f}s)", flush=True)
    print(f"  {day}: {written} rows  ({_time.time()-t0:.0f}s)")
    return written


def sync_raw_to_local(all_days: list, local_raw_dir: str) -> None:
    """拉 GCS 上已有的天，只对缺失的天查 BQ。"""
    gcs_client = gcs.Client()
    bucket = gcs_client.bucket(GCS_BUCKET)
    bq_client = None   # lazy init

    cached, missing = [], []
    for day in all_days:
        blob_name = f"{GCS_RAW_PREFIX}/{day}.ndjson"
        local     = os.path.join(local_raw_dir, FILE_TMPL.format(day=day))
        if gcs_exists(bucket, blob_name):
            cached.append(day)
            gcs_download(bucket, blob_name, local)
        else:
            missing.append(day)

    if cached:
        print(f"[GCS cache] hit {len(cached)} days: {cached[0]} .. {cached[-1]}")
    if not missing:
        print("[BQ] all days cached, skip BQ pull")
        return

    print(f"[BQ] pulling {len(missing)} missing days: {missing}")
    bq_client = bigquery.Client(project=BQ_PROJECT)
    for day in missing:
        local     = os.path.join(local_raw_dir, FILE_TMPL.format(day=day))
        blob_name = f"{GCS_RAW_PREFIX}/{day}.ndjson"
        pull_day(bq_client, day, local)
        gcs_upload(local, bucket, blob_name)   # 缓存到 GCS，下次不再拉 BQ


# ── step 2: build features ────────────────────────────────────────────────────

def build_features(raw_dir: str, art_dir: str) -> None:
    print(f"\n[features] building artifacts -> {art_dir}")
    env = {
        **os.environ,
        "TRAIN_DAYS": ",".join(TRAIN_DAYS),
        "TEST_DAYS":  ",".join(TEST_DAYS),
        "DATA_DIR":   raw_dir,
        "OUT_DIR":    art_dir,
    }
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "pipeline" / "build_features.py")],
        env=env, check=True,
    )


# ── step 3: upload artifacts → GCS latest/ ───────────────────────────────────

def upload_artifacts(art_dir: str) -> None:
    prefix = f"{GCS_ART_PREFIX}/latest"
    gcs_client = gcs.Client()
    bucket = gcs_client.bucket(GCS_BUCKET)
    print(f"\n[GCS] uploading artifacts -> gs://{GCS_BUCKET}/{prefix}/")
    for fname in ["train.npz", "test.npz", "config.json", "vocab.json"]:
        local = os.path.join(art_dir, fname)
        if os.path.exists(local):
            gcs_upload(local, bucket, f"{prefix}/{fname}")
        else:
            print(f"  WARNING: {fname} not found, skipping")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"train_days ({len(TRAIN_DAYS)}): {TRAIN_DAYS[0]} .. {TRAIN_DAYS[-1]}")
    print(f"test_days  ({len(TEST_DAYS)}):  {TEST_DAYS}")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        art_dir = os.path.join(tmpdir, "artifacts")
        os.makedirs(raw_dir); os.makedirs(art_dir)

        sync_raw_to_local(TRAIN_DAYS + TEST_DAYS, raw_dir)
        build_features(raw_dir, art_dir)
        upload_artifacts(art_dir)

    print("\ndone.")


if __name__ == "__main__":
    main()
