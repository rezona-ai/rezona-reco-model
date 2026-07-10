#!/usr/bin/env python3
"""
gRPC scoring server for TwoTowerLite coarse ranking.

Env vars:
  GCS_BUCKET                e.g. rezona-ml
  GCS_ARTIFACTS_PREFIX      e.g. two_tower_lite/artifacts  (reads latest/)
  GCS_CKPT_PREFIX           e.g. two_tower_lite/ckpts
  GCS_ITEM_VECTORS_PREFIX   e.g. two_tower_lite/item_vectors
  GRPC_PORT                 default 50051
  GRPC_WORKERS              default 4
  REFRESH_ENABLED           default true  (daily 6am refresh)

  # local dev only (GCS_BUCKET unset)
  ART_DIR                   path to artifacts/ dir (config.json, vocab.json)
  CKPT_PATH                 path to two_tower_lite_best.pt
  ITEM_VECTORS_DIR          path to dir with item_vectors.npy + game_ids.json
"""
import datetime
import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import orjson
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from two_tower_lite import TwoTowerLite

import twotower_ranker_pb2
import twotower_ranker_pb2_grpc
from encoder import UserEncoder
from item_store import ServingBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

GCS_BUCKET              = os.environ.get("GCS_BUCKET", "")
GCS_ARTIFACTS_PREFIX    = os.environ.get("GCS_ARTIFACTS_PREFIX",    "two_tower_lite/artifacts")
GCS_CKPT_PREFIX         = os.environ.get("GCS_CKPT_PREFIX",         "two_tower_lite/ckpts")
GCS_ITEM_VECTORS_PREFIX = os.environ.get("GCS_ITEM_VECTORS_PREFIX", "two_tower_lite/item_vectors")
GRPC_PORT               = int(os.environ.get("GRPC_PORT",    50051))
GRPC_WORKERS            = int(os.environ.get("GRPC_WORKERS", (os.cpu_count() or 4) * 2))
REFRESH_ENABLED         = os.environ.get("REFRESH_ENABLED", "true").lower() == "true"

_RETRY_INTERVAL_SEC = 300   # 5 min between retries waiting for today's files
_RETRY_MAX          = 12    # give up after 1 hour (12 × 5 min)


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _gcs_client():
    from google.cloud import storage as gcs_lib
    return gcs_lib.Client()


def _gcs_download(client, blob_name: str, local_path: str) -> None:
    client.bucket(GCS_BUCKET).blob(blob_name).download_to_filename(local_path)


def _gcs_exists(client, blob_name: str) -> bool:
    return client.bucket(GCS_BUCKET).blob(blob_name).exists()


def _find_ready_date() -> str:
    """Return the latest date that has both ckpt and item_vecs, falling back to earlier dates."""
    client = _gcs_client()
    blobs = client.bucket(GCS_BUCKET).list_blobs(prefix=GCS_CKPT_PREFIX + "/")
    depth = len(GCS_CKPT_PREFIX.split("/"))
    dates = sorted({
        b.name.split("/")[depth]
        for b in blobs
        if len(b.name.split("/")) > depth and b.name.split("/")[depth].startswith("20")
    }, reverse=True)
    if not dates:
        raise RuntimeError(f"No ckpt dates under gs://{GCS_BUCKET}/{GCS_CKPT_PREFIX}/")
    for date in dates:
        if (_gcs_exists(client, f"{GCS_CKPT_PREFIX}/{date}/two_tower_lite_best.pt") and
                _gcs_exists(client, f"{GCS_ITEM_VECTORS_PREFIX}/{date}/item_vectors.npy")):
            if date != dates[0]:
                log.warning("latest ckpt date=%s has no item_vecs, falling back to date=%s",
                            dates[0], date)
            return date
    raise RuntimeError(f"No date found with both ckpt and item_vecs under gs://{GCS_BUCKET}/")


def _date_ready(date: str) -> bool:
    client = _gcs_client()
    return (
        _gcs_exists(client, f"{GCS_CKPT_PREFIX}/{date}/two_tower_lite_best.pt") and
        _gcs_exists(client, f"{GCS_ITEM_VECTORS_PREFIX}/{date}/item_vectors.npy")
    )


# ── bundle construction ───────────────────────────────────────────────────────

def _build_bundle_gcs(tmp: str, date: str) -> ServingBundle:
    """Download artifacts + ckpt + item vecs for `date`, return ServingBundle."""
    client = _gcs_client()
    for fname in ["config.json", "vocab.json"]:
        _gcs_download(client, f"{GCS_ARTIFACTS_PREFIX}/latest/{fname}", f"{tmp}/{fname}")
    with open(f"{tmp}/config.json") as f:
        cfg = json.load(f)
    with open(f"{tmp}/vocab.json", "rb") as f:
        vocab = {k: {str(tk): int(iv) for tk, iv in m.items()}
                 for k, m in orjson.loads(f.read()).items()}

    ckpt_path = f"{tmp}/best.pt"
    _gcs_download(client, f"{GCS_CKPT_PREFIX}/{date}/two_tower_lite_best.pt", ckpt_path)
    model = TwoTowerLite(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    vecs_path = f"{tmp}/item_vectors.npy"
    ids_path  = f"{tmp}/game_ids.json"
    _gcs_download(client, f"{GCS_ITEM_VECTORS_PREFIX}/{date}/item_vectors.npy", vecs_path)
    _gcs_download(client, f"{GCS_ITEM_VECTORS_PREFIX}/{date}/game_ids.json",    ids_path)
    item_vecs = np.load(vecs_path).astype(np.float32, copy=False)
    with open(ids_path, "rb") as f:
        game_ids = orjson.loads(f.read())
    mapping = {int(gid): i for i, gid in enumerate(game_ids)}

    log.info("bundle ready: date=%s  games=%d  params=%.2fM",
             date, len(game_ids), sum(p.numel() for p in model.parameters()) / 1e6)
    return ServingBundle(model, UserEncoder(cfg, vocab), item_vecs, mapping, date)


def _build_bundle_local() -> ServingBundle:
    art_dir   = os.environ["ART_DIR"]
    ckpt_path = os.environ["CKPT_PATH"]
    iv_dir    = os.environ["ITEM_VECTORS_DIR"]

    with open(f"{art_dir}/config.json") as f:
        cfg = json.load(f)
    with open(f"{art_dir}/vocab.json", "rb") as f:
        vocab = {k: {str(tk): int(iv) for tk, iv in m.items()}
                 for k, m in orjson.loads(f.read()).items()}

    model = TwoTowerLite(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    item_vecs = np.load(f"{iv_dir}/item_vectors.npy").astype(np.float32)
    with open(f"{iv_dir}/game_ids.json", "rb") as f:
        game_ids = orjson.loads(f.read())
    mapping = {int(gid): i for i, gid in enumerate(game_ids)}

    return ServingBundle(model, UserEncoder(cfg, vocab), item_vecs, mapping, "local")


# ── refresh scheduling ────────────────────────────────────────────────────────

def _seconds_until_6am() -> float:
    now    = datetime.datetime.now()
    target = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


# ── gRPC servicer ─────────────────────────────────────────────────────────────

class TwoTowerRankerServicer(twotower_ranker_pb2_grpc.TwoTowerRankerServiceServicer):
    def __init__(self, bundle: ServingBundle):
        self._bundle = bundle   # atomic swap on refresh (GIL guarantees ref assignment atomicity)

    def Score(self, request, context):
        bundle   = self._bundle             # snapshot reference before doing work
        u        = request.user
        game_ids = list(request.game_ids)

        scores_arr = bundle.score(u.user_id, u.country_code, u.platform, u.app_version, game_ids)

        return twotower_ranker_pb2.TwoTowerScoreResponse(
            scores=[
                twotower_ranker_pb2.TwoTowerGameScore(game_id=gid, score=float(s))
                for gid, s in zip(game_ids, scores_arr)
            ],
            model_date=bundle.date,
        )

    def start_refresh_thread(self) -> None:
        t = threading.Thread(target=self._refresh_loop, daemon=True, name="bundle-refresh")
        t.start()

    def _refresh_loop(self) -> None:
        while True:
            wait = _seconds_until_6am()
            log.info("next bundle refresh at 06:00 (%.0fs from now)", wait)
            time.sleep(wait)

            today = datetime.date.today().isoformat()
            for attempt in range(1, _RETRY_MAX + 1):
                try:
                    if _date_ready(today):
                        with tempfile.TemporaryDirectory() as tmp:
                            new_bundle = _build_bundle_gcs(tmp, today)
                        self._bundle = new_bundle   # atomic swap
                        log.info("bundle refreshed → date=%s", today)
                        break
                    log.info("date=%s not ready yet (attempt %d/%d), retry in %ds",
                             today, attempt, _RETRY_MAX, _RETRY_INTERVAL_SEC)
                except Exception as e:
                    log.warning("refresh attempt %d/%d failed: %s", attempt, _RETRY_MAX, e)
                time.sleep(_RETRY_INTERVAL_SEC)
            else:
                log.warning("gave up refreshing to date=%s after %d attempts", today, _RETRY_MAX)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if GCS_BUCKET:
        with tempfile.TemporaryDirectory() as tmp:
            date   = _find_ready_date()
            bundle = _build_bundle_gcs(tmp, date)
    else:
        bundle = _build_bundle_local()

    servicer = TwoTowerRankerServicer(bundle)
    if REFRESH_ENABLED and GCS_BUCKET:
        servicer.start_refresh_thread()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=GRPC_WORKERS))
    twotower_ranker_pb2_grpc.add_TwoTowerRankerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    log.info("gRPC server listening  port=%d  workers=%d  date=%s",
             GRPC_PORT, GRPC_WORKERS, bundle.date)

    def _handle_sigterm(*_):
        log.info("SIGTERM received, graceful shutdown (grace=20s)")
        server.stop(grace=20)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
