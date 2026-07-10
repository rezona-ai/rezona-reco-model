#!/usr/bin/env python3
"""
Export item (game) embedding vectors from the latest trained TwoTowerLite checkpoint.

Steps:
  1. Load config.json + vocab.json from GCS artifacts/latest/
  2. Load two_tower_lite_best.pt from the latest GCS ckpt directory
  3. Query BQ ads.ads_game_info_df_view (latest dt) with published-game filters
  4. Encode item features and run model.item_vec()
  5. Save item_vectors.npy + game_ids.json to GCS

Output: gs://{GCS_BUCKET}/{GCS_ITEM_VECTORS_PREFIX}/{YYYY-MM-DD}/
          item_vectors.npy   float32 (N, 64)
          game_ids.json      list of N game_id strings (same order)

Env vars:
  GCS_BUCKET                 e.g. rezona-ml
  GCS_ARTIFACTS_PREFIX       e.g. two_tower_lite/artifacts
  GCS_CKPT_PREFIX            e.g. two_tower_lite/ckpts
  GCS_ITEM_VECTORS_PREFIX    e.g. two_tower_lite/item_vectors  (default)
  BQ_PROJECT                 default: rezonaai
  BQ_TABLE                   default: rezonaai.ads.ads_game_info_df_view
"""
import os, sys, json, math, datetime, tempfile
from pathlib import Path

import numpy as np
import torch
import orjson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from google.cloud import bigquery, storage as gcs_lib

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from two_tower_lite import TwoTowerLite

# ── config ────────────────────────────────────────────────────────────────────
GCS_BUCKET              = os.environ["GCS_BUCKET"]
GCS_ARTIFACTS_PREFIX    = os.environ["GCS_ARTIFACTS_PREFIX"]
GCS_CKPT_PREFIX         = os.environ["GCS_CKPT_PREFIX"]
GCS_ITEM_VECTORS_PREFIX = os.environ.get("GCS_ITEM_VECTORS_PREFIX", "two_tower_lite/item_vectors")
BQ_PROJECT              = os.environ.get("BQ_PROJECT", "rezonaai")
BQ_TABLE                = os.environ.get("BQ_TABLE", "rezonaai.ads.ads_game_info_df_view")

PAD, OOV = 0, 1

# ── column mapping: ads_game_info_df_view → slim format vocab keys ────────────
# (feat_name_in_table, vocab_key)
CAT_ITEM_COLS = [
    ("game_id",       "game_id"),
    ("creator_id",    "author_id"),   # creator_id in table = author_id in vocab
    ("game_category", "category"),
    ("game_tag",      "tag"),
    ("art_style",     "art_style"),
]

# table column → maps to training slim key order (must match NUM_FEATURES in build_features.py)
NUM_COL_MAP = [
    "play_cnt_td",           # g_play
    "show_cnt_td",           # g_show
    "like_cnt_td",           # g_like
    "comment_cnt_td",        # g_comment
    "remix_cnt_td",          # g_remix
    "share_cnt_td",          # g_share
    "game_version_cnt",      # g_ver
    "author_fans_cnt_td",    # a_fans
    "author_following_cnt_td",  # a_following
    "author_play_cnt_td",    # a_play
    "author_like_cnt_td",    # a_like
    "author_publish_games_td",  # a_publish
    "author_comment_cnt_td", # a_comment
    "author_remix_cnt_td",   # a_remix
    "author_share_cnt_td",   # a_share
    # game_age_days computed separately below
]


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _bucket():
    return gcs_lib.Client().bucket(GCS_BUCKET)

def gcs_download(blob_name, local_path):
    _bucket().blob(blob_name).download_to_filename(local_path)

def gcs_upload(local_path, blob_name):
    _bucket().blob(blob_name).upload_from_filename(local_path)
    print(f"  → gs://{GCS_BUCKET}/{blob_name}  ({os.path.getsize(local_path)/1e6:.1f}MB)")

def latest_ckpt_date():
    """Find the most recent date directory under GCS_CKPT_PREFIX."""
    blobs = gcs_lib.Client().bucket(GCS_BUCKET).list_blobs(prefix=GCS_CKPT_PREFIX + "/")
    dates = sorted({b.name.split("/")[len(GCS_CKPT_PREFIX.split("/"))]
                    for b in blobs
                    if len(b.name.split("/")) > len(GCS_CKPT_PREFIX.split("/"))
                    and b.name.split("/")[len(GCS_CKPT_PREFIX.split("/"))].startswith("20")})
    if not dates:
        raise RuntimeError(f"No checkpoint dates found under gs://{GCS_BUCKET}/{GCS_CKPT_PREFIX}/")
    return dates[-1]


# ── feature encoding helpers ──────────────────────────────────────────────────

def norm_token(v):
    if v is None: return None
    s = str(v).strip()
    return s if s else None

def enc_cat(vocab, vname, raw):
    t = norm_token(raw)
    if t is None: return OOV
    return vocab.get(vname, {}).get(t, OOV)

def encode_numeric(row, num_mean, num_std):
    vals = [float(row.get(col)) if row.get(col) is not None else None for col in NUM_COL_MAP]
    out = []
    for j, v in enumerate(vals):
        if v is None:
            out.append(0.0)
        else:
            lv = math.copysign(math.log1p(abs(v)), v)
            out.append(float((lv - num_mean[j]) / num_std[j]))
    return out


# ── BQ pull ───────────────────────────────────────────────────────────────────

_BQ_COLS = ", ".join([
    "game_id", "creator_id", "game_category", "game_tag", "art_style",
    "play_cnt_td", "show_cnt_td", "like_cnt_td", "comment_cnt_td",
    "remix_cnt_td", "share_cnt_td", "game_version_cnt",
    "author_fans_cnt_td", "author_following_cnt_td", "author_play_cnt_td",
    "author_like_cnt_td", "author_publish_games_td", "author_comment_cnt_td",
    "author_remix_cnt_td", "author_share_cnt_td",
    "create_ts",
])

def pull_games():
    bq = bigquery.Client(project=BQ_PROJECT)
    sql = f"""
    SELECT {_BQ_COLS}
    FROM `{BQ_TABLE}`
    WHERE dt = (SELECT MAX(dt) FROM `{BQ_TABLE}`)
      AND public       = TRUE
      AND is_deleted   = FALSE
      AND game_status  = 'published'
      AND nsfw         = FALSE
    """
    print(f"[BQ] {BQ_TABLE}")
    arrow_table = bq.query(sql).to_arrow()
    rows = arrow_table.to_pylist()
    print(f"  {len(rows):,} qualifying games")
    return rows


# ── visualization ─────────────────────────────────────────────────────────────

def _plot_emb(vecs: np.ndarray, N: int, D: int, out_path: str) -> None:
    RNG      = np.random.default_rng(42)
    SAMPLE_N = 50_000
    PAIR_N   = 100_000

    norms    = np.linalg.norm(vecs, axis=1)
    dim_mean = vecs.mean(axis=0)
    dim_std  = vecs.std(axis=0)

    # PCA 2D on sample
    idx  = RNG.choice(N, size=min(SAMPLE_N, N), replace=False)
    samp = vecs[idx]
    pca2 = PCA(n_components=2, random_state=42)
    xy   = pca2.fit_transform(samp)
    var2 = pca2.explained_variance_ratio_

    # full PCA for cumulative variance
    pca_full = PCA(n_components=min(D, len(samp)), random_state=42)
    pca_full.fit(samp)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n90  = int(np.searchsorted(cum_var, 0.90)) + 1
    n95  = int(np.searchsorted(cum_var, 0.95)) + 1

    # cosine similarity on random pairs
    ia = RNG.choice(N, size=PAIR_N, replace=True)
    ib = RNG.choice(N, size=PAIR_N, replace=True)
    a, b = vecs[ia].astype(np.float32), vecs[ib].astype(np.float32)
    cos  = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Item Embedding Analysis  (N={N:,}, D={D})", fontsize=14)

    # 1. L2 norm
    ax = fig.add_subplot(2, 3, 1)
    ax.hist(norms, bins=100, color="#4C72B0", edgecolor="none", alpha=0.85)
    ax.set_title(f"L2 Norm  mean={norms.mean():.2f}  std={norms.std():.2f}")
    ax.set_xlabel("||v||₂"); ax.set_ylabel("count"); ax.grid(alpha=0.3)

    # 2. per-dim mean
    ax = fig.add_subplot(2, 3, 2)
    ax.bar(range(D), dim_mean, color="#55A868", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Per-dimension Mean"); ax.set_xlabel("dim"); ax.grid(alpha=0.3)

    # 3. per-dim std
    ax = fig.add_subplot(2, 3, 3)
    ax.bar(range(D), dim_std, color="#C44E52", alpha=0.85)
    ax.set_title("Per-dimension Std"); ax.set_xlabel("dim"); ax.grid(alpha=0.3)

    # 4. PCA 2D scatter (density colored)
    ax = fig.add_subplot(2, 3, 4)
    h, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=200)
    h = gaussian_filter(h, sigma=1)
    xi = np.clip(np.searchsorted(xe[1:], xy[:, 0]), 0, h.shape[0]-1)
    yi = np.clip(np.searchsorted(ye[1:], xy[:, 1]), 0, h.shape[1]-1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=h[xi, yi], s=0.4,
                    cmap="viridis", alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="density")
    ax.set_title(f"PCA 2D  var={var2[0]:.1%}+{var2[1]:.1%}  n={min(SAMPLE_N,N):,}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.2)

    # 5. cosine similarity distribution
    ax = fig.add_subplot(2, 3, 5)
    ax.hist(cos, bins=100, color="#8172B2", edgecolor="none", alpha=0.85)
    ax.set_title(f"Cosine Sim (random pairs)  mean={cos.mean():.3f}  std={cos.std():.3f}")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("count"); ax.grid(alpha=0.3)

    # 6. PCA cumulative variance
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(range(1, len(cum_var)+1), cum_var, linewidth=1.5)
    ax.axhline(0.90, color="red",    linestyle="--", linewidth=1, label=f"90% @ {n90}d")
    ax.axhline(0.95, color="orange", linestyle="--", linewidth=1, label=f"95% @ {n95}d")
    ax.set_title("PCA Cumulative Variance")
    ax.set_xlabel("# components"); ax.set_ylabel("cumulative var")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. load artifacts
        print(f"\n[1/4] loading artifacts  gs://{GCS_BUCKET}/{GCS_ARTIFACTS_PREFIX}/latest/")
        for fname in ["config.json", "vocab.json"]:
            gcs_download(f"{GCS_ARTIFACTS_PREFIX}/latest/{fname}", f"{tmpdir}/{fname}")

        with open(f"{tmpdir}/config.json") as f:
            cfg = json.load(f)
        vocab_raw = orjson.loads(open(f"{tmpdir}/vocab.json", "rb").read())
        vocab = {k: {str(tk): int(iv) for tk, iv in m.items()} for k, m in vocab_raw.items()}
        num_mean = np.array(cfg["numeric"]["mean"], dtype=np.float64)
        num_std  = np.array(cfg["numeric"]["std"],  dtype=np.float64)

        # 2. load checkpoint
        ckpt_date = latest_ckpt_date()
        ckpt_blob = f"{GCS_CKPT_PREFIX}/{ckpt_date}/two_tower_lite_best.pt"
        print(f"\n[2/4] loading checkpoint  gs://{GCS_BUCKET}/{ckpt_blob}")
        ckpt_path = f"{tmpdir}/best.pt"
        gcs_download(ckpt_blob, ckpt_path)
        model = TwoTowerLite(cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # 3. pull games
        print(f"\n[3/4] pulling games from BQ...")
        rows = pull_games()
        if not rows:
            print("  no games found, exit")
            return

        # 4. encode + compute item vectors
        print(f"\n[4/4] encoding {len(rows):,} games and computing item vectors...")
        game_ids = []
        cat_lists  = {f"cat__{feat}": [] for _, feat in [(c[0], c[1]) for c in CAT_ITEM_COLS]}
        # rebuild with proper key names matching model input
        cat_buf = {f"cat__{vname}": [] for _, vname in CAT_ITEM_COLS}
        num_buf = []

        # model item_cats uses feat names from config cat_single
        # input keys must be cat__{feat} where feat is the original feat name in CAT_SINGLE
        # e.g. cat__game_id, cat__author_id, cat__game_category, cat__game_tag, cat__art_style
        feat_key_map = {
            "game_id":       "cat__game_id",
            "creator_id":    "cat__author_id",
            "game_category": "cat__game_category",
            "game_tag":      "cat__game_tag",
            "art_style":     "cat__art_style",
        }

        INFER_BS = 50_000
        cat_bufs = {v: [] for v in feat_key_map.values()}
        num_buf  = []

        for row in rows:
            r = dict(row)
            game_ids.append(int(r.get("game_id") or 0))
            for col, vname in CAT_ITEM_COLS:
                cat_bufs[feat_key_map[col]].append(enc_cat(vocab, vname, r.get(col)))
            num_buf.append(encode_numeric(r, num_mean, num_std))

        all_vecs = []
        n = len(game_ids)
        for start in range(0, n, INFER_BS):
            end = min(start + INFER_BS, n)
            batch = {k: torch.tensor(v[start:end], dtype=torch.long) for k, v in cat_bufs.items()}
            batch["num"] = torch.tensor(num_buf[start:end], dtype=torch.float32)
            with torch.no_grad():
                all_vecs.append(model.item_vec(batch).numpy().astype(np.float32))
            if (start // INFER_BS) % 5 == 0:
                print(f"  {end:,}/{n:,}")

        item_vecs = np.concatenate(all_vecs, axis=0)
        N, D = item_vecs.shape
        print(f"  shape: {item_vecs.shape}  (dtype={item_vecs.dtype})")

        # 5. visualize
        print("\n[5/6] generating embedding analysis plot...")
        plot_path = f"{tmpdir}/emb_analysis.png"
        _plot_emb(item_vecs, N, D, plot_path)

        # 6. upload
        today      = datetime.date.today().isoformat()
        out_prefix = f"{GCS_ITEM_VECTORS_PREFIX}/{today}"

        vecs_path = f"{tmpdir}/item_vectors.npy"
        ids_path  = f"{tmpdir}/game_ids.json"
        np.save(vecs_path, item_vecs)
        with open(ids_path, "wb") as f:
            f.write(orjson.dumps(game_ids))

        print(f"\n[GCS] uploading → gs://{GCS_BUCKET}/{out_prefix}/")
        gcs_upload(vecs_path, f"{out_prefix}/item_vectors.npy")
        gcs_upload(ids_path,  f"{out_prefix}/game_ids.json")
        gcs_upload(plot_path, f"{out_prefix}/emb_analysis.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
