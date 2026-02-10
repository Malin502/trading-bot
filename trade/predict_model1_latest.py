# scripts/predict_model1_latest.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
ENSEMBLE_METHOD = "weighted_mean"

# weighted_mean時の重み設定
RECENCY_WEIGHT_POWER = 1.0
FOLD_QUALITY_WEIGHT = True
MIN_FOLD_QUALITY = 0.1

# 不確実性でポジション縮小
UNCERTAINTY_PERCENTILE = 0.80  # 上位20%のみ縮小（バランス型）
MIN_POSITION_SCALE = 0.5       # 縮小時は50%に（中程度）

# スコアリング（利益重視）
SCORING_METHOD = "ret_only"  # "ret_only" | "utility" | "sharpe_adj" | "cost_aware" | "simple"
RISK_AVERSION = 0.5          # リターンとリスクのバランス（推奨: 0.3-0.7）
COST_BPS = 5.0               # 取引コスト（実態に合わせる）
SLIPPAGE_BPS = 3.0           # スリッページ

# スコア設定
EPS = 1e-6
SCORE_CLIP = None  # 例: 50.0 / Noneで無効

# 予測安定化
RET_WINSOR_LOW_Q = 0.02
RET_WINSOR_HIGH_Q = 0.98

# リスク予測崩壊の検知（銘柄間差がほぼ無い場合）
RISK_SPREAD_MIN = 0.02
RISK_CV_MIN = 0.03

# 保有継続ボーナス（回転率削減）
HOLDING_BONUS_SCORE = 0.02  # 前日保有銘柄へのスコア加算（2%程度）

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


def _build_model_from_state_dict(in_dim: int, state_dict: Dict[str, torch.Tensor]) -> Model1ResidualMLP:
    """
    学習時設定に依存せず、state_dictからwidth/depthを復元する。
    """
    w_key = "trunk.0.weight"
    if w_key not in state_dict:
        raise KeyError(f"missing key in state_dict: {w_key}")
    width = int(state_dict[w_key].shape[0])
    depth = sum(1 for k in state_dict.keys() if k.startswith("trunk.") and k.endswith("net.0.weight"))
    if depth <= 0:
        raise ValueError("could not infer residual depth from state_dict")
    return Model1ResidualMLP(in_dim=in_dim, width=width, depth=depth, dropout=0.0)


def _fold_quality_weight(fold_dir: Path, min_weight: float = 0.1) -> float:
    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.exists():
        return 1.0
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sharpe = float((metrics.get("val_backtest") or {}).get("sharpe_daily", 0.0))
        rankic = float((metrics.get("val_backtest") or {}).get("rankic_mean", 0.0))
        # バックテスト系の最小限スコア。負値は下限へ丸める。
        quality = max(min_weight, 1.0 + sharpe + 0.5 * rankic)
        return float(quality)
    except Exception as e:
        print(f"[WARN] fold quality parse failed for {fold_dir.name}: {e}")
        return 1.0


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
    use_risk: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["pred_risk_abs"] = out["pred_risk"].abs()
    total_cost = (cost_bps + slippage_bps) / 10000.0

    if not use_risk:
        score = out["pred_ret"] - total_cost
    else:
        if method == "ret_only":
            score = out["pred_ret"] - total_cost
        elif method == "utility":
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


def _is_risk_collapsed(df_pred: pd.DataFrame) -> bool:
    risk = df_pred["pred_risk"].to_numpy(dtype=np.float64)
    if len(risk) < 5:
        return False
    q95, q05 = np.nanquantile(risk, [0.95, 0.05])
    spread = float(q95 - q05)
    mean = float(np.nanmean(risk))
    std = float(np.nanstd(risk))
    cv = std / (abs(mean) + 1e-8)
    collapsed = (spread < RISK_SPREAD_MIN) or (cv < RISK_CV_MIN)
    print(f"[INFO] risk spread={spread:.6f}, cv={cv:.6f}, collapsed={collapsed}")
    return collapsed


def load_previous_holdings(latest_day: pd.Timestamp, topk: int = 5) -> set:
    """前日の保有銘柄を取得（回転率削減用）"""
    try:
        prev_day = latest_day - pd.Timedelta(days=1)
        # 営業日でない可能性があるので、過去10日分探索
        for i in range(1, 11):
            check_date = latest_day - pd.Timedelta(days=i)
            prev_file = OUT_DIR / f"rank_{check_date.date()}.csv"
            if prev_file.exists():
                df_prev = pd.read_csv(prev_file)
                if "score_adj" in df_prev.columns:
                    holdings = set(df_prev.nlargest(topk, "score_adj")["ticker"].tolist())
                elif "score" in df_prev.columns:
                    holdings = set(df_prev.nlargest(topk, "score")["ticker"].tolist())
                else:
                    holdings = set(df_prev.head(topk)["ticker"].tolist())
                return holdings
    except Exception as e:
        print(f"[WARN] Could not load previous holdings: {e}")
    return set()


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

        model = _build_model_from_state_dict(in_dim=X_final.shape[1], state_dict=state_dict)
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
        recency_weights = {fid: float((i + 1) ** RECENCY_WEIGHT_POWER) for i, fid in enumerate(fold_ids_sorted)}
        quality_weights = {fid: 1.0 for fid in fold_ids_sorted}
        if FOLD_QUALITY_WEIGHT:
            fold_dir_map = {int(p.name.split("_")[1]): p for p in fold_dirs}
            quality_weights = {
                fid: _fold_quality_weight(fold_dir_map[fid], min_weight=MIN_FOLD_QUALITY)
                for fid in fold_ids_sorted
            }

        weights = {fid: recency_weights[fid] * quality_weights[fid] for fid in fold_ids_sorted}
        print(f"[INFO] fold weights: {weights}")

        df_pred_all["weight"] = df_pred_all["fold_id"].map(weights).astype(float)
        df_pred_all["weighted_ret"] = df_pred_all["pred_ret"] * df_pred_all["weight"]
        df_pred_all["weighted_risk"] = df_pred_all["pred_risk"] * df_pred_all["weight"]

        grouped = df_pred_all.groupby("ticker")
        sum_w = grouped["weight"].sum().rename("sum_w")
        wret = grouped["weighted_ret"].sum().rename("sum_wret")
        wrisk = grouped["weighted_risk"].sum().rename("sum_wrisk")
        std_df = grouped.agg(
            pred_ret_std=("pred_ret", "std"),
            pred_risk_std=("pred_risk", "std"),
        )
        df_pred_mean = pd.concat([sum_w, wret, wrisk, std_df], axis=1).reset_index()
        df_pred_mean["pred_ret"] = df_pred_mean["sum_wret"] / df_pred_mean["sum_w"].clip(lower=1e-8)
        df_pred_mean["pred_risk"] = df_pred_mean["sum_wrisk"] / df_pred_mean["sum_w"].clip(lower=1e-8)
        df_pred_mean = df_pred_mean.drop(columns=["sum_w", "sum_wret", "sum_wrisk"])

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

    # 極端予測の抑制（クロスセクションwinsorize）
    if RET_WINSOR_LOW_Q is not None and RET_WINSOR_HIGH_Q is not None:
        lo, hi = df_pred_mean["pred_ret"].quantile([RET_WINSOR_LOW_Q, RET_WINSOR_HIGH_Q]).tolist()
        df_pred_mean["pred_ret_raw"] = df_pred_mean["pred_ret"]
        df_pred_mean["pred_ret"] = df_pred_mean["pred_ret"].clip(lo, hi)
        print(f"[INFO] pred_ret winsorize: q{RET_WINSOR_LOW_Q:.2f}={lo:.6f}, q{RET_WINSOR_HIGH_Q:.2f}={hi:.6f}")

    risk_collapsed = _is_risk_collapsed(df_pred_mean)

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
        use_risk=not risk_collapsed,
    )
    out["score_method_used"] = "ret_only" if risk_collapsed else SCORING_METHOD
    
    # 保有継続ボーナスを追加（回転率削減）
    prev_holdings = load_previous_holdings(pd.Timestamp(latest_day), topk=5)
    if prev_holdings:
        print(f"[INFO] Previous holdings: {prev_holdings}")
        out["holding_bonus"] = out["ticker"].apply(lambda x: HOLDING_BONUS_SCORE if x in prev_holdings else 0.0)
        out["score"] = out["score"] + out["holding_bonus"]
    else:
        out["holding_bonus"] = 0.0
        print("[WARN] No previous holdings found, skipping holding bonus")
    
    out["score_adj"] = out["score"] * out["position_scale"]
    out = out.sort_values("score_adj", ascending=False).reset_index(drop=True)

    # 保存
    out_path = OUT_DIR / f"rank_{latest_day}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
