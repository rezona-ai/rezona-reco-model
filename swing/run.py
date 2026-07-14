#!/usr/bin/env python3
"""
Swing i2i daily job: BQ pull → slim GCS cache → Swing algorithm → eval → upload artifacts.

GCS layout (independent from two_tower_lite):
  swing/slim/YYYY-MM-DD.ndjson.gz          per-day user interaction slim cache (train)
  swing/eval/YYYY-MM-DD.ndjson.gz          per-day eval cache (positive target + triggers)
  swing/artifacts/YYYY-MM-DD/
      i2i_index.json                       game_id → Top-K neighbors (full)
      i2i_index_simple.json                game_id → [neighbor_game_id, ...]
      i2i_stats.json                       build statistics + eval results

Slim format (one line per BQ impression row):
  {"uid": int, "games": [{"gid": int, "label": int, "pt": int}, ...]}

Eval format (one line per positive target record with at least one trigger):
  {"uid": int, "triggers": [gid, ...], "target": gid}

  Trigger: recent_games entry with interaction_label > 0 OR play_time > TRIGGER_PT_MS
  Target:  shown game with context_info interaction OR playing_time > TARGET_PT_MS

Env vars:
  GCS_BUCKET              required
  GCS_SLIM_PREFIX         default: swing/slim
  GCS_EVAL_PREFIX         default: swing/eval
  GCS_ARTIFACTS_PREFIX    default: swing/artifacts
  BQ_PROJECT              default: rezonaai
  BQ_TABLE                default: rezonaai.datalake.reco_labeled_log_flat
  BQ_PARTITION_COL        default: dt
  BQ_WHERE                default: context_info.realshow = TRUE
  BQ_TIMEOUT_SEC          default: 1800
  SWING_DAYS              default: 18  (SWING_DAYS-1 train days + 1 eval day)
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
GCS_EVAL_PREFIX      = os.environ.get("GCS_EVAL_PREFIX",      "swing/eval")
GCS_ARTIFACTS_PREFIX = os.environ.get("GCS_ARTIFACTS_PREFIX", "swing/artifacts")
BQ_PROJECT           = os.environ.get("BQ_PROJECT",           "rezonaai")
BQ_TABLE             = os.environ.get("BQ_TABLE",             "rezonaai.datalake.reco_labeled_log_flat")
BQ_PART_COL          = os.environ.get("BQ_PARTITION_COL",     "dt")
BQ_WHERE             = os.environ.get("BQ_WHERE",             "context_info.realshow = TRUE").strip()
BQ_TIMEOUT           = int(os.environ.get("BQ_TIMEOUT_SEC",   1800))
SWING_DAYS           = int(os.environ.get("SWING_DAYS",       18))
PROGRESS_N           = 50_000

# play_time thresholds — NOTE: different units per field
TRIGGER_PT_S  = 10        # recent_games[].play_time unit is seconds
TARGET_PT_MS  = 10_000    # context_info.playing_time unit is milliseconds

EVAL_K_VALUES = (10, 20, 50)


# ── date window ───────────────────────────────────────────────────────────────

def resolve_days() -> tuple[list[str], str]:
    """
    Returns (train_days, eval_day).
    eval_day  = yesterday (last 1 day, held out for evaluation only)
    train_days = the SWING_DAYS-1 days before that (used to build the index)
    Total window = SWING_DAYS days.
    """
    today    = datetime.date.today()
    eval_dt  = today - datetime.timedelta(days=1)
    start_dt = eval_dt - datetime.timedelta(days=SWING_DAYS - 1)
    train_days = [(start_dt + datetime.timedelta(days=i)).isoformat()
                  for i in range(SWING_DAYS - 1)]
    eval_day = eval_dt.isoformat()
    print(f"today={today}  train=[{train_days[0]}, {train_days[-1]}]  "
          f"eval={eval_day}  ({SWING_DAYS-1} train + 1 eval)")
    return train_days, eval_day


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


# ── BQ: train slim pull ───────────────────────────────────────────────────────

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


# ── BQ: eval pull ─────────────────────────────────────────────────────────────

def pull_day_eval(bq: bigquery.Client, day: str, local_gz: str) -> int:
    """
    Pull eval data for one day.

    For each impression:
    - triggers: recent_games entries that are positive
                (interaction_label > 0 OR play_time > TRIGGER_PT_MS)
    - target:   game_info.game_id, only if context_info is positive
                (any of like/follow/forward/remix not null, OR playing_time > TARGET_PT_MS)

    Only records with at least one trigger AND a positive target are written.
    Records with no recent_games are skipped.

    Output format: {"uid": int, "triggers": [gid, ...], "target": gid}

    Note: recent_games[].play_time is in seconds (threshold: TRIGGER_PT_S).
          context_info.playing_time is in milliseconds (threshold: TARGET_PT_MS).
    """
    extra = f" AND ({BQ_WHERE})" if BQ_WHERE else ""
    sql = f"""
        SELECT
            user_info.user_id         AS uid,
            user_info.recent_games    AS recent_games,
            game_info.game_id         AS target_gid,
            context_info.like         AS ctx_like,
            context_info.follow       AS ctx_follow,
            context_info.forward      AS ctx_forward,
            context_info.remix        AS ctx_remix,
            context_info.playing_time AS ctx_pt
        FROM `{BQ_TABLE}`
        WHERE {BQ_PART_COL} = @day{extra}
          AND user_info.user_id IS NOT NULL
          AND user_info.recent_games IS NOT NULL
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("day", "DATE", day)],
        job_timeout_ms=BQ_TIMEOUT * 1000,
    ))

    written = 0
    skipped_no_target = 0
    skipped_no_trigger = 0
    t0 = time.time()

    with gzip.open(local_gz, "wb") as f:
        for row in job.result():
            uid        = row["uid"]
            target_gid = row["target_gid"]
            if not uid or not target_gid:
                continue

            # check if target (shown game) is positive
            ctx_pt = int(row.get("ctx_pt") or 0)
            target_positive = bool(
                row.get("ctx_like") or
                row.get("ctx_follow") or
                row.get("ctx_forward") or
                row.get("ctx_remix") or
                ctx_pt > TARGET_PT_MS
            )
            if not target_positive:
                skipped_no_target += 1
                continue

            # collect positive trigger games from recent_games
            triggers = []
            for g in (row["recent_games"] or []):
                gid = g.get("game_id")
                if not gid:
                    continue
                lbl = int(g.get("interaction_label") or 0)
                pt  = int(g.get("play_time") or 0)
                if lbl > 0 or pt > TRIGGER_PT_S:
                    triggers.append(int(gid))

            if not triggers:
                skipped_no_trigger += 1
                continue

            f.write(orjson.dumps({
                "uid":      int(uid),
                "triggers": triggers,
                "target":   int(target_gid),
            }) + b"\n")
            written += 1
            if written % PROGRESS_N == 0:
                print(f"  ... {written:,} rows ({time.time()-t0:.0f}s)", flush=True)

    size_mb = os.path.getsize(local_gz) / 1e6
    print(f"  {day}: {written:,} eval records  {size_mb:.1f}MB  ({time.time()-t0:.0f}s)")
    print(f"  skipped: {skipped_no_target:,} (no positive target)  "
          f"{skipped_no_trigger:,} (no trigger games)")
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


def ensure_eval_in_gcs(eval_day: str) -> str:
    """Ensure eval data for eval_day is cached in GCS. Returns local blob name."""
    bucket    = _bucket()
    blob_name = f"{GCS_EVAL_PREFIX}/{eval_day}.ndjson.gz"
    if _exists(bucket, blob_name):
        print(f"[eval cache] hit {eval_day}")
        return blob_name

    print(f"[eval] pulling eval day {eval_day} from BQ...")
    bq = bigquery.Client(project=BQ_PROJECT)
    with tempfile.TemporaryDirectory() as tmp:
        local_gz = os.path.join(tmp, f"{eval_day}.ndjson.gz")
        n = pull_day_eval(bq, eval_day, local_gz)
        if n > 0:
            _upload(local_gz, bucket, blob_name)
    return blob_name


# ── download helpers ──────────────────────────────────────────────────────────

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


def download_eval_day(eval_day: str, tmp: str) -> str:
    bucket = _bucket()
    local  = os.path.join(tmp, f"eval_{eval_day}.ndjson.gz")
    _download(bucket, f"{GCS_EVAL_PREFIX}/{eval_day}.ndjson.gz", local)
    print(f"  ↓ eval/{eval_day}.ndjson.gz  ({os.path.getsize(local)/1e6:.1f}MB)")
    return local


# ── eval ──────────────────────────────────────────────────────────────────────

def run_eval(index: dict, eval_path: str) -> dict:
    """
    Compute Recall@K for the i2i index.

    For each eval record:
    - Collect top-K neighbors from all trigger games (union).
    - Hit if target game_id appears in the retrieved set.

    Coverage = fraction of eval samples where at least one trigger has neighbors.
    """
    hits     = {k: 0 for k in EVAL_K_VALUES}
    total    = 0
    covered  = 0

    open_fn = gzip.open if eval_path.endswith(".gz") else open
    with open_fn(eval_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            rec      = orjson.loads(line)
            triggers = rec.get("triggers") or []
            target   = rec.get("target")
            if not triggers or not target:
                continue
            total += 1

            if any(t in index for t in triggers):
                covered += 1

            for k in EVAL_K_VALUES:
                retrieved = set()
                for gid in triggers:
                    for nbr in (index.get(gid) or [])[:k]:
                        retrieved.add(nbr["game_id"])
                if target in retrieved:
                    hits[k] += 1

    results = {
        "eval_samples":    total,
        "index_coverage":  round(covered / max(total, 1), 4),
    }
    for k in EVAL_K_VALUES:
        results[f"recall@{k}"] = round(hits[k] / max(total, 1), 6)

    print(f"  eval_samples={total:,}  index_coverage={results['index_coverage']:.3f}")
    for k in EVAL_K_VALUES:
        print(f"  Recall@{k:<3d} = {results[f'recall@{k}']:.4f}  "
              f"({hits[k]:,}/{total:,})")
    return results


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
    train_days, eval_day = resolve_days()
    today = datetime.date.today().isoformat()
    t0    = time.time()

    print("\n[Step 1] Ensuring train slim cache in GCS...")
    ensure_slim_in_gcs(train_days)

    print("\n[Step 2] Ensuring eval cache in GCS...")
    ensure_eval_in_gcs(eval_day)

    with tempfile.TemporaryDirectory() as tmp:
        print("\n[Step 3] Downloading train slim files...")
        paths = download_slim_days(train_days, tmp)

        print("\n[Pass 1] Building user-game interaction map...")
        user_games, n_records, n_pos = build_user_game_map_from_slim(paths)
        print(f"  unique users: {len(user_games):,}  records: {n_records:,}  pos interactions: {n_pos:,}")

        print("\n[Pass 2] Computing Swing co-occurrence...")
        co_occ, item_user_cnt, qualified = build_co_occurrence(user_games, MIN_ITEM_USERS)

        print("\n[Step 4] Building top-K neighbor index...")
        index = build_index(co_occ, user_games, qualified, item_user_cnt, ALPHA, TOP_K)
        print(f"  games with neighbors: {len(index):,}")
        nbr_counts = [len(v) for v in index.values()]
        avg_nbr    = sum(nbr_counts) / len(nbr_counts) if nbr_counts else 0
        print(f"  avg neighbors/game: {avg_nbr:.1f}  max: {max(nbr_counts) if nbr_counts else 0}")

        print(f"\n[Step 5] Evaluating on eval day {eval_day}...")
        eval_local = download_eval_day(eval_day, tmp)
        eval_results = run_eval(index, eval_local)

        elapsed = time.time() - t0
        stats = {
            "date":       today,
            "train_days": train_days,
            "eval_day":   eval_day,
            "n_records":               n_records,
            "n_positive_interactions": n_pos,
            "n_users_with_history":    len(user_games),
            "n_qualified_items":       len(qualified),
            "n_co_occurrence_pairs":   len(co_occ),
            "n_items_in_index":        len(index),
            "avg_neighbors":           round(avg_nbr, 2),
            "eval":        eval_results,
            "params": {
                "SWING_DAYS":     SWING_DAYS,
                "MIN_ITEM_USERS": MIN_ITEM_USERS,
                "MIN_CO_USERS":   MIN_CO_USERS,
                "TOP_K":          TOP_K,
                "ALPHA":          ALPHA,
                "MAX_PAIR_USERS": MAX_PAIR_USERS,
                "USER_GAME_CAP":  USER_GAME_CAP,
                "TRIGGER_PT_S":  TRIGGER_PT_S,
                "TARGET_PT_MS":  TARGET_PT_MS,
            },
            "elapsed_sec": round(elapsed, 1),
        }
        upload_artifacts(index, stats, today, tmp)

    print(f"\ndone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
