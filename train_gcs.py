#!/usr/bin/env python3
"""
Job 2 — 从 GCS 下载 artifacts → 训练 → 上传 ckpt 到 GCS。

Env vars:
  GCS_BUCKET              e.g. rezona-ml
  GCS_ARTIFACTS_PREFIX    e.g. two_tower_lite/artifacts  (会读 latest/ 子目录)
  GCS_CKPT_PREFIX         e.g. two_tower_lite/ckpts      (会写 YYYY-MM-DD/ 子目录)
  EPOCHS                  default 2
  BS                      default 4096
  LR                      default 3e-3
  EVAL_EVERY              default 50
"""
import os, sys, json, copy, csv, time, datetime, tempfile, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.cloud import storage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from two_tower_lite import TwoTowerLite
from metrics import auc, gauc

OOV = 1


# ── GCS helpers ───────────────────────────────────────────────────────────────

def gcs_download_dir(bucket_name: str, prefix: str, local_dir: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs  = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise RuntimeError(f"No files found at gs://{bucket_name}/{prefix}")
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        filename = os.path.basename(blob.name)
        if not filename:
            continue
        dest = os.path.join(local_dir, filename)
        blob.download_to_filename(dest)
        print(f"  downloaded {filename} ({os.path.getsize(dest)/1e6:.1f}MB)")


def gcs_upload_file(local_path: str, bucket_name: str, blob_name: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  uploaded {os.path.basename(local_path)} ({size_mb:.1f}MB) "
          f"-> gs://{bucket_name}/{blob_name}")


# ── data loading ──────────────────────────────────────────────────────────────

def load_split(art_dir: str, name: str):
    d = np.load(os.path.join(art_dir, f"{name}.npz"))
    t = {}
    for k in d.files:
        arr = d[k]
        if k in ("label", "playing_time"):
            t[k] = torch.tensor(arr, dtype=torch.float32)
        elif k == "uid_group":
            t[k] = torch.tensor(arr, dtype=torch.long)
        elif arr.dtype == np.float32:
            t[k] = torch.tensor(arr, dtype=torch.float32)
        else:
            t[k] = torch.tensor(arr.astype(np.int64), dtype=torch.long)
    return t, len(t["label"])


def gather_batch(data, idx):
    return {k: v[idx] for k, v in data.items() if k != "uid_group"}


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _scores(model, data, n, bs=8192):
    model.eval()
    s = np.empty(n, dtype=np.float32)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n))
        s[i:i + len(idx)] = torch.sigmoid(model(gather_batch(data, idx))).cpu().numpy()
    model.train()
    return s


def evaluate_seg(model, data, n, new_mask, bs=8192):
    s = _scores(model, data, n, bs)
    y = data["label"].numpy()
    g = data["uid_group"].numpy()
    res = {}
    ga, cov = gauc(y, s, g)
    res["auc"], res["gauc"], res["cov"] = auc(y, s), ga, cov
    for seg, m in [("new", new_mask), ("old", ~new_mask)]:
        if m.sum() > 0:
            gg, _ = gauc(y[m], s[m], g[m])
            res[f"auc_{seg}"], res[f"gauc_{seg}"] = auc(y[m], s[m]), gg
        else:
            res[f"auc_{seg}"], res[f"gauc_{seg}"] = float("nan"), float("nan")
    return res


# ── training loop ─────────────────────────────────────────────────────────────

def train(art_dir: str, ckpt_dir: str, epochs: int, bs: int, lr: float, eval_every: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    with open(os.path.join(art_dir, "config.json")) as f:
        cfg = json.load(f)

    print("loading train split...")
    train_data, n_tr = load_split(art_dir, "train")
    print("loading test split...")
    test_data,  n_te = load_split(art_dir, "test")

    train_data = {k: v.to(device) for k, v in train_data.items()}
    test_data  = {k: v.to(device) for k, v in test_data.items()}

    new_mask = (test_data["cat__user_id"].cpu().numpy() == OOV)
    print(f"train rows={n_tr}  test rows={n_te}  "
          f"pos(train)={train_data['label'].mean().item():.4f}  "
          f"pos(test)={test_data['label'].mean().item():.4f}")
    print(f"test new users(OOV)={new_mask.mean():.3f}")

    model = TwoTowerLite(cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    total_steps = epochs * math.ceil(n_tr / bs)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=0)

    keys = ["step", "loss", "train_auc", "train_gauc",
            "test_auc", "test_gauc", "test_auc_new", "test_gauc_new",
            "test_auc_old", "test_gauc_old"]
    hist = {k: [] for k in keys}
    best = {"gauc": -1, "step": 0, "state": None, "auc": 0}
    buf_y, buf_s, buf_g = [], [], []
    step = 0
    t0 = time.time()

    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            idx   = perm[i:i + bs]
            y     = train_data["label"][idx]
            logit = model(gather_batch(train_data, idx))
            loss  = lossf(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            scheduler.step()
            step += 1
            buf_y.append(y.detach().cpu().numpy())
            buf_s.append(torch.sigmoid(logit).detach().cpu().numpy())
            buf_g.append(train_data["uid_group"][idx].cpu().numpy())

            if step % eval_every == 0:
                ty = np.concatenate(buf_y)
                ts = np.concatenate(buf_s)
                tg = np.concatenate(buf_g)
                tr_auc = auc(ty, ts); tr_g, _ = gauc(ty, ts, tg)
                buf_y, buf_s, buf_g = [], [], []
                r = evaluate_seg(model, test_data, n_te, new_mask)
                vals = [step, loss.item(), tr_auc, tr_g, r["auc"], r["gauc"],
                        r["auc_new"], r["gauc_new"], r["auc_old"], r["gauc_old"]]
                for k, v in zip(keys, vals): hist[k].append(v)
                if r["gauc"] > best["gauc"]:
                    best = {"gauc": r["gauc"], "auc": r["auc"], "step": step,
                            "state": copy.deepcopy(model.state_dict())}
                print(f"ep{ep} step{step:5d} loss={loss.item():.4f} | "
                      f"train AUC {tr_auc:.4f} GAUC {tr_g:.4f} | "
                      f"TEST AUC {r['auc']:.4f} GAUC {r['gauc']:.4f} | "
                      f"old {r['gauc_old']:.4f} new {r['gauc_new']:.4f} | "
                      f"{time.time()-t0:.0f}s")

        r = evaluate_seg(model, test_data, n_te, new_mask)
        print(f"== epoch {ep} == TEST AUC {r['auc']:.4f} GAUC {r['gauc']:.4f} "
              f"| old {r['auc_old']:.4f}/{r['gauc_old']:.4f} "
              f"| new {r['auc_new']:.4f}/{r['gauc_new']:.4f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, "two_tower_lite.pt")
    torch.save(model.state_dict(), final_path)

    best_path = None
    if best["state"] is not None:
        best_path = os.path.join(ckpt_dir, "two_tower_lite_best.pt")
        torch.save(best["state"], best_path)
        print(f"BEST ckpt @ step {best['step']}: AUC {best['auc']:.4f} GAUC {best['gauc']:.4f}")

    curves_path = os.path.join(ckpt_dir, "curves.csv")
    with open(curves_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(hist))
        for row in zip(*hist.values()): w.writerow(row)

    plot_path = os.path.join(ckpt_dir, "curves.png")
    _plot_curves(hist, best["step"], plot_path)

    return final_path, best_path, curves_path, plot_path


def _plot_curves(hist: dict, best_step: int, out_path: str) -> None:
    steps = hist["step"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in [
        (axes[0], "auc",  "AUC"),
        (axes[1], "gauc", "GAUC"),
    ]:
        ax.plot(steps, hist[f"train_{metric}"], label="train", linewidth=1.5)
        ax.plot(steps, hist[f"test_{metric}"],  label="test",  linewidth=1.5)
        if best_step and best_step in steps:
            bx = steps.index(best_step)
            ax.axvline(x=best_step, color="gray", linestyle="--", linewidth=1, label=f"best@{best_step}")
            ax.scatter([best_step], [hist[f"test_{metric}"][bx]], color="red", zorder=5)
        ax.set_title(title); ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved curves plot -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    bucket      = os.environ["GCS_BUCKET"]
    art_prefix  = os.environ["GCS_ARTIFACTS_PREFIX"]
    ckpt_prefix = os.environ["GCS_CKPT_PREFIX"]
    epochs      = int(os.environ.get("EPOCHS", 2))
    bs          = int(os.environ.get("BS", 4096))
    lr          = float(os.environ.get("LR", 3e-3))
    eval_every  = int(os.environ.get("EVAL_EVERY", 50))

    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir  = os.path.join(tmpdir, "artifacts")
        ckpt_dir = os.path.join(tmpdir, "ckpt")

        # 1. 从 GCS 下载 artifacts（同 region，免费）
        print(f"\n[1/3] downloading artifacts from gs://{bucket}/{art_prefix}/latest/")
        gcs_download_dir(bucket, f"{art_prefix}/latest", art_dir)

        # 2. 训练
        print(f"\n[2/3] training (epochs={epochs} bs={bs} lr={lr})")
        final_path, best_path, curves_path, plot_path = train(
            art_dir, ckpt_dir, epochs, bs, lr, eval_every
        )

        # 3. 上传 ckpt（按日期）
        today = datetime.date.today().isoformat()
        dated_prefix = f"{ckpt_prefix}/{today}"
        print(f"\n[3/3] uploading ckpts to gs://{bucket}/{dated_prefix}/")
        for local in [final_path, best_path, curves_path, plot_path]:
            if local and os.path.exists(local):
                gcs_upload_file(local, bucket, f"{dated_prefix}/{os.path.basename(local)}")

    print("\ndone.")


if __name__ == "__main__":
    main()
