# scripts/predict_model1_latest.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# あなたの既存データセット生成コード（clean_table / load_universe_table 等）
from src.preprocessing.DatasetBuilder import (
    DatasetSettings,
    load_universe_table,
    clean_table,
)

# モデル定義（Model1ResidualMLP）
from src.model1.model import Model1ResidualMLP


# ============================
# Settings (WRITE HERE)
# ============================

# 特徴量（推論用：当日のみ）parquet
FEATURES_PARQUET = Path("features/universe/features_1h_for_model1_latest_1d.parquet")

# 学習成果物
ARTIFACTS_ROOT = Path("artifacts/model1")

# 直近何foldでアンサンブルするか
N_RECENT_FOLDS = 5

# アンサンブル方法: "mean" | "median" | "weighted_mean"
ENSEMBLE_METHOD = "mean"

# 不確実性でポジション縮小
UNCERTAINTY_PERCENTILE = 0.75
MIN_POSITION_SCALE = 0.3

# スコアリング（利益重視）
SCORING_METHOD = "utility"  # "utility" | "sharpe_adj" | "cost_aware" | "simple"
RISK_AVERSION = 1.0
COST_BPS = 5.0
SLIPPAGE_BPS = 3.0

# スコア設定
EPS = 1e-6
SCORE_CLIP = None  # 例: 50.0 / Noneで無効

# 出力先
OUT_DIR = Path("artifacts/model1/inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================
# Helpers
# ============================

def _list_fold_dirs(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.glob("fold_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )


def _select_recent_folds(root: Path, n_recent: int) -> List[Path]:
    fold_dirs = _list_fold_dirs(root)
    if not fold_dirs:
        raise SystemExit(f"No fold dirs under: {root}")
    if len(fold_dirs) < n_recent:
        print(f"[WARN] only {len(fold_dirs)} folds found, using all")
        n_recent = len(fold_dirs)
    return fold_dirs[-n_recent:]


def _load_fold_artifacts(fold_dir: Path):
    model_path = fold_dir / "model.pt"
    scaler_path = fold_dir / "scaler.pkl"
    feat_path = fold_dir / "feature_cols.json"
    scale_cols_path = fold_dir / "scale_cols.json"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    feature_cols: List[str] = json.loads(feat_path.read_text(encoding="utf-8"))
    scale_cols: Optional[List[str]] = None
    if scale_cols_path.exists():
        scale_cols = json.loads(scale_cols_path.read_text(encoding="utf-8"))

    scaler = joblib.load(scaler_path)
    state_dict = torch.load(model_path, map_location="cpu")

    return feature_cols, scale_cols, scaler, state_dict


def _build_fold_inputs(
    df_today: pd.DataFrame,
    feature_cols: List[str],
    scaler,
    scale_cols: Optional[List[str]] = None,
) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df_today.columns]
    if missing:
        raise ValueError(f"Missing feature cols in today's df: {missing[:10]} ... total={len(missing)}")

    if scale_cols is None:
        passthrough_cols = [c for c in feature_cols if c.endswith("_sin") or c.endswith("_cos")]
        scale_cols = [c for c in feature_cols if c not in passthrough_cols]
    else:
        passthrough_cols = [c for c in feature_cols if c not in scale_cols]

    X_scale = df_today[scale_cols].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_scale).astype(np.float32)

    if passthrough_cols:
        X_pass = df_today[passthrough_cols].to_numpy(dtype=np.float32)
        X_tmp = np.concatenate([X_scaled, X_pass], axis=1)
        tmp_cols = scale_cols + passthrough_cols
        col_to_i = {c: i for i, c in enumerate(tmp_cols)}
        idx = [col_to_i[c] for c in feature_cols]
        X_final = X_tmp[:, idx].astype(np.float32)
    else:
        X_final = X_scaled

    return X_final


def add_score(
    df: pd.DataFrame,
    eps: float = 1e-6,
    score_clip: Optional[float] = None,
    method: str = "simple",
    risk_aversion: float = 1.0,
    cost_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> pd.DataFrame:
    out = df.copy()
    out["pred_risk_abs"] = out["pred_risk"].abs()
    total_cost = (cost_bps + slippage_bps) / 10000.0

    if method == "utility":
        score = out["pred_ret"] - risk_aversion * (out["pred_risk_abs"] ** 2) - total_cost
    elif method == "sharpe_adj":
        score = (out["pred_ret"] - total_cost) / (out["pred_risk_abs"] + eps) - 0.5 * out["pred_risk_abs"]
    elif method == "cost_aware":
        score = (out["pred_ret"] - total_cost) / (out["pred_risk_abs"] + eps)
    else:
        score = out["pred_ret"] / (out["pred_risk_abs"] + eps)

    out["score"] = score
    if score_clip is not None and score_clip > 0:
        out["score"] = out["score"].clip(-score_clip, score_clip)
    return out


# ============================
# Main
# ============================

def main():
    # --- 特徴量ロード（あなたの既存コードを利用） ---
    s = DatasetSettings(universe_parquet=FEATURES_PARQUET, require_labels=False)
    df = load_universe_table(s)

    # clean_table は require_labels=False でラベル不要にできる
    df, _ = clean_table(df, s, feature_cols=None)  # feature_cols推論（ただし学習契約は artifacts を使う）

    # indexがDatetimeIndexで "Datetime" 名になっている前提（あなたのread関数に準拠）
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")

    # --- 最新日だけに絞る（当日15:00の行群） ---
    latest_day = df.index.date.max()
    df_today = df.loc[df.index.date == latest_day].copy()
    if len(df_today) == 0:
        raise SystemExit("No rows for latest day")

    # ticker列必須
    if s.id_col not in df_today.columns:
        raise ValueError(f"'{s.id_col}' column not found")

    # --- foldアンサンブル ---
    fold_dirs = _select_recent_folds(ARTIFACTS_ROOT, N_RECENT_FOLDS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split("_")[1])
        feature_cols, scale_cols, scaler, state_dict = _load_fold_artifacts(fold_dir)

        X_final = _build_fold_inputs(df_today, feature_cols, scaler, scale_cols)

        model = Model1ResidualMLP(in_dim=X_final.shape[1], width=256, depth=4, dropout=0.10)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        with torch.no_grad():
            xt = torch.from_numpy(X_final).to(device)
            pred_ret, pred_risk = model(xt)
            pr = pred_ret.detach().cpu().numpy()
            pk = pred_risk.detach().cpu().numpy()

        df_pred_fold = pd.DataFrame({
            "ticker": df_today[s.id_col].astype(str).to_numpy(),
            "fold_id": fold_id,
            "pred_ret": pr,
            "pred_risk": pk,
        })
        all_preds.append(df_pred_fold)

    if not all_preds:
        raise SystemExit("No valid predictions from any fold")

    df_pred_all = pd.concat(all_preds, ignore_index=True)

    if ENSEMBLE_METHOD == "mean":
        df_pred_mean = df_pred_all.groupby("ticker").agg({
            "pred_ret": "mean",
            "pred_risk": "mean",
        }).reset_index()

        df_pred_std = df_pred_all.groupby("ticker").agg({
            "pred_ret": "std",
            "pred_risk": "std",
        }).rename(columns={
            "pred_ret": "pred_ret_std",
            "pred_risk": "pred_risk_std",
        }).reset_index()

        df_pred_mean = df_pred_mean.merge(df_pred_std, on="ticker")

    elif ENSEMBLE_METHOD == "weighted_mean":
        fold_ids_sorted = sorted(df_pred_all["fold_id"].unique())
        weights = {fid: (i + 1) for i, fid in enumerate(fold_ids_sorted)}
        df_pred_all["weight"] = df_pred_all["fold_id"].map(weights)

        df_pred_mean = (
            df_pred_all
            .groupby("ticker")
            .apply(lambda g: pd.Series({
                "pred_ret": (g["pred_ret"] * g["weight"]).sum() / g["weight"].sum(),
                "pred_risk": (g["pred_risk"] * g["weight"]).sum() / g["weight"].sum(),
                "pred_ret_std": g["pred_ret"].std(),
                "pred_risk_std": g["pred_risk"].std(),
            }))
            .reset_index()
        )

    elif ENSEMBLE_METHOD == "median":
        df_pred_mean = df_pred_all.groupby("ticker").agg({
            "pred_ret": "median",
            "pred_risk": "median",
        }).reset_index()

        df_pred_std = df_pred_all.groupby("ticker").agg({
            "pred_ret": "std",
            "pred_risk": "std",
        }).rename(columns={
            "pred_ret": "pred_ret_std",
            "pred_risk": "pred_risk_std",
        }).reset_index()

        df_pred_mean = df_pred_mean.merge(df_pred_std, on="ticker")

    else:
        raise ValueError(f"Unknown ENSEMBLE_METHOD: {ENSEMBLE_METHOD}")

    df_pred_mean["pred_uncertainty"] = (
        df_pred_mean["pred_ret_std"] / (df_pred_mean["pred_ret"].abs() + 1e-3)
    )

    unc = df_pred_mean["pred_uncertainty"].replace([np.inf, -np.inf], np.nan)
    unc_threshold = unc.quantile(UNCERTAINTY_PERCENTILE)
    df_pred_mean["position_scale"] = np.where(
        unc > unc_threshold,
        MIN_POSITION_SCALE,
        1.0,
    )

    out = df_pred_mean.copy()
    out["datetime"] = pd.Timestamp(latest_day)
    out = add_score(
        out,
        eps=EPS,
        score_clip=SCORE_CLIP,
        method=SCORING_METHOD,
        risk_aversion=RISK_AVERSION,
        cost_bps=COST_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    out["score_adj"] = out["score"] * out["position_scale"]
    out = out.sort_values("score_adj", ascending=False).reset_index(drop=True)

    # 保存
    out_path = OUT_DIR / f"rank_{latest_day}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
