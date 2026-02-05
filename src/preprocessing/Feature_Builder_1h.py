# build_features_1h_for_ae.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.market_data.CodeExtractor import extract_topix_codes

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREP_ROOT = PROJECT_ROOT / "data" / "preprocessing"
FEATURES_ROOT = PROJECT_ROOT / "features"
MARKET_TICKER = "1306.T"

REQ_COLS = ["Open", "High", "Low", "Close", "Volume"]

# rolling window sizes (in bars)
K_VOL = 20      # for return volatility
K_VREL = 20     # for volume relative
EPS = 1e-12


def _read_preprocessed_ohlcv(ticker: str) -> pd.DataFrame:
    fp = PREP_ROOT / ticker / "intraday_1h_preprocessed.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing preprocessed parquet: {fp}")
    df = pd.read_parquet(fp)
    if "Datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("Datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: index is not DatetimeIndex")
    df = df.sort_index()
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns: {missing}")
    return df[REQ_COLS].copy()


def _hour_sin_cos(index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    # hour as continuous value: hour + minute/60
    hour = index.hour + index.minute / 60.0
    angle = 2.0 * math.pi * (hour / 24.0)
    return pd.Series(np.sin(angle), index=index), pd.Series(np.cos(angle), index=index)


def _compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    max_oc = np.maximum(o.values, c.values)
    min_oc = np.minimum(o.values, c.values)

    body = (c - o) / (o + EPS)
    range_ = (h - l) / (o + EPS)
    upper_wick = (h - pd.Series(max_oc, index=df.index)) / (o + EPS)
    lower_wick = (pd.Series(min_oc, index=df.index) - l) / (o + EPS)

    # 1h log-return
    r = np.log((c + EPS) / (c.shift(1) + EPS))
    r_abs = r.abs()

    # volume features (volume can be 0; allow it)
    v = np.log(df["Volume"].astype(float) + 1.0)
    v_sma = v.rolling(K_VREL, min_periods=K_VREL).mean()
    v_rel = v - v_sma

    # volatility of returns
    vol = r.rolling(K_VOL, min_periods=K_VOL).std()

    hour_sin, hour_cos = _hour_sin_cos(df.index)

    out = pd.DataFrame(
        {
            "body": body,
            "range": range_,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "r": r,
            "r_abs": r_abs,
            "v_log": v,
            "v_rel": v_rel,
            "vol": vol,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
        },
        index=df.index,
    )
    return out


def _align_and_add_market_diffs(
    feat_stock: pd.DataFrame, feat_mkt: pd.DataFrame
) -> pd.DataFrame:
    need = ["r", "vol", "v_rel"]
    for col in need:
        if col not in feat_mkt.columns:
            raise ValueError(f"Market features missing: {col}")

    # --- NEW: align market to stock timestamps ---
    mkt = feat_mkt[need].copy()
    mkt = mkt.reindex(feat_stock.index)

    # --- NEW: forward-fill ONLY within the same date (prevents cross-day leakage) ---
    # If market has missing at some timestamps, fill with earlier timestamp of the same day
    mkt["date"] = mkt.index.date
    mkt[need] = mkt.groupby("date")[need].ffill()
    mkt = mkt.drop(columns=["date"])

    joined = feat_stock.join(
        mkt.rename(columns={"r": "r_mkt", "vol": "vol_mkt", "v_rel": "v_rel_mkt"}),
        how="inner",
    )

    # If still NaNs remain (e.g., day-start missing), drop those rows
    joined = joined.dropna(subset=["r_mkt", "vol_mkt", "v_rel_mkt"])

    joined["rel_r"] = joined["r"] - joined["r_mkt"]
    joined["rel_vol"] = joined["vol"] - joined["vol_mkt"]
    joined["rel_v"] = joined["v_rel"] - joined["v_rel_mkt"]

    return joined.drop(columns=["r_mkt", "vol_mkt", "v_rel_mkt"])


def _fit_scaler(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Return per-column (mean, std) fitted on training slice."""
    scaler: Dict[str, Tuple[float, float]] = {}
    for col in train_df.columns:
        mu = float(train_df[col].mean())
        sd = float(train_df[col].std(ddof=0))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        scaler[col] = (mu, sd)
    return scaler


def _apply_scaler(df: pd.DataFrame, scaler: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (mu, sd) in scaler.items():
        if col in out.columns:
            out[col] = (out[col] - mu) / sd
    return out


def _save_features(ticker: str, df_feat: pd.DataFrame) -> None:
    out_dir = FEATURES_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "features_1h_for_ae.parquet"
    df_feat.to_parquet(fp)
    
def _fit_global_scaler(feat_by_ticker: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    # concat train slices across tickers
    trains = []
    for tkr, feat in feat_by_ticker.items():
        n = len(feat)
        if n < 200:
            continue
        split = int(n * 0.7)
        trains.append(feat.iloc[:split])
    if not trains:
        raise ValueError("No sufficient data to fit global scaler.")
    train_all = pd.concat(trains, axis=0)
    return _fit_scaler(train_all)


def main():
    tickers = list(extract_topix_codes())[:100]

    market_fp = PREP_ROOT / MARKET_TICKER / "intraday_1h_preprocessed.parquet"
    if not market_fp.exists():
        raise FileNotFoundError(f"Market ticker preprocessed file not found: {market_fp}")

    df_mkt_ohlcv = _read_preprocessed_ohlcv(MARKET_TICKER)
    feat_mkt = _compute_candle_features(df_mkt_ohlcv).dropna()

    # --- NEW: first pass build features for all tickers (unscaled) ---
    feat_map: Dict[str, pd.DataFrame] = {}
    for i, tkr in enumerate(tickers, 1):
        df_ohlcv = _read_preprocessed_ohlcv(tkr)
        feat = _compute_candle_features(df_ohlcv).dropna()
        feat = _align_and_add_market_diffs(feat, feat_mkt)
        feat_map[tkr] = feat
        print(f"[PASS1 {i:03d}/{len(tickers)}] {tkr} rows={len(feat)}")

    # --- NEW: fit one global scaler ---
    global_scaler = _fit_global_scaler(feat_map)

    # --- NEW: second pass apply global scaler and save ---
    for i, tkr in enumerate(tickers, 1):
        feat = feat_map[tkr]
        feat_scaled = _apply_scaler(feat, global_scaler)
        _save_features(tkr, feat_scaled)
        print(f"[PASS2 {i:03d}/{len(tickers)}] OK {tkr} rows={len(feat_scaled)} cols={feat_scaled.shape[1]}")

    print("Done.")



if __name__ == "__main__":
    main()
