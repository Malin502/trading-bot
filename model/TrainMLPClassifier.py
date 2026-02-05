from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 現在のディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from MLPClassifier import MLPClassifier, sigmoid, compute_metrics, compute_pos_weight, find_best_threshold


DATA_ROOT = Path("../data/datasets/intraday_1h_ae")
LATENT_FP = DATA_ROOT / "latent32.parquet"
FOLDS_FP = DATA_ROOT / "folds.json"

OUT_DIR = Path("../models/mlp_latent32")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_col: str = "y"):
        self.x = df[feature_cols].to_numpy(dtype=np.float32)
        self.y = df[label_col].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.int64)


def load_fold_manifest(k: int, split: str) -> pd.DataFrame:
    fp = DATA_ROOT / f"fold_{k:03d}_{split}.parquet"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_parquet(fp)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if "end_ts" in df.columns:
        df["end_ts"] = pd.to_datetime(df["end_ts"])

    return df


def merge_latent(manifest: pd.DataFrame, latent_df: pd.DataFrame) -> pd.DataFrame:
    keys = []
    for k in ["ticker", "date", "end_ts"]:
        if k in manifest.columns and k in latent_df.columns:
            keys.append(k)
    if len(keys) >= 2:
        return manifest.merge(latent_df, on=keys, how="inner")
    return manifest.merge(latent_df, on=["ticker", "date"], how="inner")


@torch.no_grad()
def predict(model: nn.Module, dl: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, logits = [], []
    for xb, yb in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        out = model(xb)
        logits.append(out.detach().cpu().numpy())
        ys.append(yb.numpy())
    y_true = np.concatenate(ys, axis=0)
    y_logit = np.concatenate(logits, axis=0)
    y_prob = sigmoid(y_logit)
    return y_true, y_prob


def train_one_fold(
    fold_k: int,
    latent_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    patience: int = 5,
) -> dict:
    tr_m = load_fold_manifest(fold_k, "train")
    va_m = load_fold_manifest(fold_k, "val")
    te_m = load_fold_manifest(fold_k, "test")

    tr = merge_latent(tr_m, latent_df)
    va = merge_latent(va_m, latent_df)
    te = merge_latent(te_m, latent_df)

    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError(f"Fold {fold_k}: empty split after merge: train={len(tr)}, val={len(va)}, test={len(te)}")

    ds_tr = LatentDataset(tr, feature_cols)
    ds_va = LatentDataset(va, feature_cols)
    ds_te = LatentDataset(te, feature_cols)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MLPClassifier(in_dim=len(feature_cols), hidden=[128, 64], dropout=dropout).to(DEVICE)

    y_train = tr["y"].to_numpy(dtype=np.int64)
    pos_weight = torch.tensor(compute_pos_weight(y_train), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).float()

            logit = model(xb)
            loss = loss_fn(logit, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = total / max(n, 1)

        y_true_va, y_prob_va = predict(model, dl_va)
        val_metrics = compute_metrics(y_true_va, y_prob_va, thr=0.5)

        print(
            f"[fold {fold_k:03d}] ep={ep:03d} train_loss={train_loss:.5f} "
            f"val_logloss={val_metrics['logloss']:.5f} val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["logloss"] < best_val - 1e-6:
            best_val = val_metrics["logloss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- Practical: choose threshold on VAL, then freeze it for TEST ----
        y_true_tr, y_prob_tr = predict(model, dl_tr)
        y_true_va, y_prob_va = predict(model, dl_va)
        y_true_te, y_prob_te = predict(model, dl_te)

        best_thr, best_val_metrics_at_thr = find_best_threshold(
            y_true_va, y_prob_va, metric="f1"
        )

        train_metrics = compute_metrics(y_true_tr, y_prob_tr, thr=best_thr)
        val_metrics = compute_metrics(y_true_va, y_prob_va, thr=best_thr)
        test_metrics = compute_metrics(y_true_te, y_prob_te, thr=best_thr)

        res = {
            "fold": fold_k,
            "best_threshold_from_val": float(best_thr),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            # 参考：最適閾値でのval指標（同じだけど保存しておくと分かりやすい）
            "val_at_best_thr": best_val_metrics_at_thr,
            "train_rows": int(len(tr)),
            "val_rows": int(len(va)),
            "test_rows": int(len(te)),
            "pos_weight": float(pos_weight.item()),
        }

    return res


def main():
    if not LATENT_FP.exists():
        raise FileNotFoundError(LATENT_FP)
    if not FOLDS_FP.exists():
        raise FileNotFoundError(FOLDS_FP)

    latent = pd.read_parquet(LATENT_FP)

    if "date" in latent.columns:
        latent["date"] = pd.to_datetime(latent["date"]).dt.tz_localize(None)
    if "end_ts" in latent.columns:
        latent["end_ts"] = pd.to_datetime(latent["end_ts"])

    feature_cols = [f"z{j:02d}" for j in range(32)]
    for c in feature_cols + ["y", "ticker", "date"]:
        if c not in latent.columns:
            raise ValueError(f"latent32.parquet missing column: {c}")

    folds_json = json.loads(FOLDS_FP.read_text(encoding="utf-8"))
    n_folds = len(folds_json["folds"])

    all_results = []
    for k in range(n_folds):
        res = train_one_fold(k, latent, feature_cols)
        all_results.append(res)

    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
