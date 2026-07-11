#!/usr/bin/env python3
"""
Swing i2i daily job: BQ pull → slim GCS cache → Swing algorithm → upload artifacts.

GCS layout (independent from two_tower_lite):
  swing/slim/YYYY-MM-DD.ndjson.gz          per-day user interaction slim cache
  swing/artifacts/YYYY-MM-DD/
      i2i_index.json                       game_id → Top-K neighbors (full)
      i2i_index_simple.json                game_id → [neighbor_game_id, ...]
      i2i_stats.json                       build statistics

Slim format (one line per BQ impression row):
  {"uid": int, "games": [{"gid": int, "label": int, "pt": int}, ...]}

Env vars:
  GCS_BUCKET              required
  GCS_SLIM_PREFIX         default: swing/slim
  GCS_ARTIFACTS_PREFIX    default: swing/artifacts
  BQ_PROJECT              default: rezonaai
  BQ_TABLE                default: rezonaai.datalake.reco_labeled_log_flat
  BQ_PARTITION_COL        default: dt
  BQ_WHERE                default: context_info.realshow = TRUE
  BQ_TIMEOUT_SEC          default: 1800
  SWING_DAYS              default: 18
"""
import datetime, gzip, math, os, sys, tempfile, time
from pathlib import Path

from google.cloud import bigquery
from google.cloud import storage as gcs_lib
import orjson

sys.path.insert(0, str(Path(__file__).parent))
from build_i2i import (
    interaction_weight,
    build_user_game_map_from_slim,
    build_co_occurrence,
    build_index,
    MIN_ITEM_USERS, MIN_CO_USERS, TOP_K, ALPHA, MAX_PAIR_USERS, USER_GAME_CAP,
)

# ── config ────────────────────────────────────────────────────────────────────
GCS_BUCKET           = os.environ["GCS_BUCKET"]
GCS_SLIM_PREFIX      = os.environ.get("GCS_SLIM_PREFIX",      "swing/slim")
GCS_ARTIFACTS_PREFIX = os.environ.get("GCS_ARTIFACTS_PREFIX", "swing/artifacts")
BQ_PROJECT           = os.environ.get("BQ_PROJECT",           "rezonaai")
BQ_TABLE             = os.environ.get("BQ_TABLE",             "rezonaai.datalake.reco_labeled_log_flat")
BQ_PART_COL          = os.environ.get("BQ_PARTITION_COL",     "dt")
BQ_WHERE             = os.environ.get("BQ_WHERE",             "context_info.realshow = TRUE").strip()
BQ_TIMEOUT           = int(os.environ.get("BQ_TIMEOUT_SEC",   1800))
SWING_DAYS           = int(os.environ.get("SWING_DAYS",       18))
PROGRESS_N           = 50_000


# ── date window ───────────────────────────────────────────────────────────────

def resolve_days() -> list[str]:
    today    = datetime.date.today()
    end_dt   = today - datetime.timedelta(days=1)          # yesterday
    start_dt = end_dt - datetime.timedelta(days=SWING_DAYS - 1)
    days = [(start_dt + datetime.timedelta(days=i)).isoformat() for i in range(SWING_DAYS)]
    print(f"today={today}  window=[{days[0]}, {days[-1]}]  ({SWING_DAYS} days)")
    return days


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _bucket():
    return gcs_lib.Client().bucket(GCS_BUCKET)

def _exists(bucket, blob: str) -> bool:
    return bucket.blob(blob).exists()

def _upload(local: str, bucket, blob: str) -> None:
    bucket.blob(blob).upload_from_filename(local)
    print(f"  → gs://{GCS_BUCKET}/{blob}  ({os.path.getsize(local)/1e6:.1f}MB)")

def _download(bucket, blob: str, local: str) -> None:
    bucket.blob(blob).download_to_filename(local)


# ── BQ pull ───────────────────────────────────────────────────────────────────

def pull_day_slim(bq: bigquery.Client, day: str, local_gz: str) -> int:
    """Pull uid + recent_games for one day from BQ, write swing slim gzip ndjson."""
    extra = f" AND ({BQ_WHERE})" if BQ_WHERE else ""
    sql = f"""
        SELECT
            user_info.user_id      AS uid,
            user_info.recent_games AS games
        FROM `{BQ_TABLE}`
        WHERE {BQ_PART_COL} = @day{extra}
          AND user_info.user_id IS NOT NULL
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("day", "DATE", day)],
        job_timeout_ms=BQ_TIMEOUT * 1000,
    ))

    written = 0
    t0 = time.time()
    with gzip.open(local_gz, "wb") as f:
        for row in job.result():
            uid = row["uid"]
            if not uid:
                continue
            games = []
            for g in (row["games"] or []):
                gid = g.get("game_id")
                if not gid:
                    continue
                games.append({
                    "gid":   int(gid),
                    "label": int(g.get("interaction_label") or 0),
                    "pt":    int(g.get("play_time") or 0),
                })
            if not games:
                continue
            f.write(orjson.dumps({"uid": int(uid), "games": games}) + b"\n")
            written += 1
            if written % PROGRESS_N == 0:
                print(f"  ... {written:,} rows ({time.time()-t0:.0f}s)", flush=True)

    print(f"  {day}: {written:,} rows  {os.path.getsize(local_gz)/1e6:.1f}MB  ({time.time()-t0:.0f}s)")
    return written


# ── slim cache ────────────────────────────────────────────────────────────────

def ensure_slim_in_gcs(days: list[str]) -> None:
    bucket  = _bucket()
    missing = [d for d in days if not _exists(bucket, f"{GCS_SLIM_PREFIX}/{d}.ndjson.gz")]

    hit = len(days) - len(missing)
    if hit:
        print(f"[slim cache] hit {hit} days")
    if not missing:
        print("[slim] all days cached"); return

    print(f"[slim] pulling {len(missing)} days from BQ: {missing}")
    bq = bigquery.Client(project=BQ_PROJECT)
    with tempfile.TemporaryDirectory() as tmp:
        for day in missing:
            local_gz  = os.path.join(tmp, f"{day}.ndjson.gz")
            blob_name = f"{GCS_SLIM_PREFIX}/{day}.ndjson.gz"
            pull_day_slim(bq, day, local_gz)
            _upload(local_gz, bucket, blob_name)


# ── download slim days for algorithm ─────────────────────────────────────────

def download_slim_days(days: list[str], tmp: str) -> list[str]:
    bucket = _bucket()
    paths  = []
    print(f"[download] fetching {len(days)} slim files...")
    for day in days:
        local = os.path.join(tmp, f"{day}.ndjson.gz")
        _download(bucket, f"{GCS_SLIM_PREFIX}/{day}.ndjson.gz", local)
        paths.append(local)
        print(f"  ↓ {day}.ndjson.gz  ({os.path.getsize(local)/1e6:.1f}MB)")
    return paths


# ── upload artifacts ──────────────────────────────────────────────────────────

def upload_artifacts(index: dict, stats: dict, today: str, tmp: str) -> None:
    bucket = _bucket()

    def _write_upload(obj, filename: str) -> None:
        local = os.path.join(tmp, filename)
        with open(local, "wb") as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
        for prefix in [f"{GCS_ARTIFACTS_PREFIX}/{today}", f"{GCS_ARTIFACTS_PREFIX}/latest"]:
            _upload(local, bucket, f"{prefix}/{filename}")

    print(f"\n[GCS] uploading artifacts → gs://{GCS_BUCKET}/{GCS_ARTIFACTS_PREFIX}/{today}/")
    _write_upload({str(k): v for k, v in index.items()}, "i2i_index.json")
    _write_upload({str(k): [n["game_id"] for n in v] for k, v in index.items()},
                  "i2i_index_simple.json")
    _write_upload(stats, "i2i_stats.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    days  = resolve_days()
    today = datetime.date.today().isoformat()
    t0    = time.time()

    print("\n[Step 1] Ensuring slim cache in GCS...")
    ensure_slim_in_gcs(days)

    with tempfile.TemporaryDirectory() as tmp:
        print("\n[Step 2] Downloading slim files...")
        paths = download_slim_days(days, tmp)

        print("\n[Pass 1] Building user-game interaction map...")
        user_games, n_records, n_pos = build_user_game_map_from_slim(paths)
        print(f"  unique users: {len(user_games):,}  records: {n_records:,}  pos interactions: {n_pos:,}")

        print("\n[Pass 2] Computing Swing co-occurrence...")
        co_occ, item_user_cnt, qualified = build_co_occurrence(user_games, MIN_ITEM_USERS)

        print("\n[Step 3] Building top-K neighbor index...")
        index = build_index(co_occ, item_user_cnt, ALPHA, TOP_K)
        print(f"  games with neighbors: {len(index):,}")
        nbr_counts = [len(v) for v in index.values()]
        avg_nbr = sum(nbr_counts) / len(nbr_counts) if nbr_counts else 0
        print(f"  avg neighbors/game: {avg_nbr:.1f}  max: {max(nbr_counts) if nbr_counts else 0}")

        elapsed = time.time() - t0
        stats = {
            "date": today,
            "days": days,
            "n_records": n_records,
            "n_positive_interactions": n_pos,
            "n_users_with_history": len(user_games),
            "n_qualified_items": len(qualified),
            "n_co_occurrence_pairs": len(co_occ),
            "n_items_in_index": len(index),
            "avg_neighbors": round(avg_nbr, 2),
            "params": {
                "SWING_DAYS":    SWING_DAYS,
                "MIN_ITEM_USERS": MIN_ITEM_USERS,
                "MIN_CO_USERS":   MIN_CO_USERS,
                "TOP_K":          TOP_K,
                "ALPHA":          ALPHA,
                "MAX_PAIR_USERS": MAX_PAIR_USERS,
                "USER_GAME_CAP":  USER_GAME_CAP,
            },
            "elapsed_sec": round(elapsed, 1),
        }
        upload_artifacts(index, stats, today, tmp)

    print(f"\ndone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
