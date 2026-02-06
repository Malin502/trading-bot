from __future__ import annotations

import sys

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.market_data.CodeExtractor import extract_topix_codes


EPS = 1e-8


# ---------------------------
# Settings (EDIT HERE)
# ---------------------------

@dataclass
class Settings:
    # 入力（前処理済み）データのルート
    prep_root: Path = Path(PROJECT_ROOT / "data/preprocessing")

    # 出力（特徴量）データのルート
    out_root: Path = Path(PROJECT_ROOT / "features")

    # 市場コード（同期インデックス）
    market_ticker: str = "1306.T"

    # 何銘柄処理するか（0=全TOPIX、例: 50なら先頭50銘柄）
    limit_tickers: int = 0

    # 全銘柄を結合したuniverse parquetも保存するか
    save_universe: bool = True

    # ラベル(y_ret,y_risk)を作成するか（学習用途ならTrue）
    make_labels: bool = True

    # 15:00バーを「その日の決定時刻」として切り出す
    decision_hour: int = 15

    # rolling window（1時間足）
    windows: Tuple[int, ...] = (8, 24, 56)

    # timezone
    tz: str = "Asia/Tokyo"


# ---------------------------
# Feature Builder (Model①)
# ---------------------------

@dataclass
class FeatureConfig:
    tz: str
    windows: Tuple[int, ...]
    decision_hour: int
    add_calendar: bool = True
    add_market: bool = True
    add_relative: bool = True
    make_labels: bool = True


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.copy()


def _normalize_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy().sort_index()
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    df.index = idx
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    df["logret_cc"] = np.log((c + EPS) / (c.shift(1) + EPS))
    df["logret_oc"] = np.log((c + EPS) / (o + EPS))

    df["range"] = (h - l) / (c + EPS)
    df["body"] = (c - o) / (c + EPS)
    df["upper_wick"] = (h - np.maximum(o, c)) / (c + EPS)
    df["lower_wick"] = (np.minimum(o, c) - l) / (c + EPS)

    df["v_log"] = np.log(v + 1.0)
    return df


def _rolling_features_leak_safe(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
    """
    IMPORTANT: shift(1) -> rolling()
    """
    df = df.copy()
    for w in windows:
        df[f"ret_std_{w}"] = df["logret_cc"].shift(1).rolling(w).std()
        df[f"ret_mean_{w}"] = df["logret_cc"].shift(1).rolling(w).mean()

        df[f"range_mean_{w}"] = df["range"].shift(1).rolling(w).mean()
        df[f"range_std_{w}"] = df["range"].shift(1).rolling(w).std()

        vol_mean = df["Volume"].shift(1).rolling(w).mean()
        df[f"v_rel_{w}"] = df["Volume"] / (vol_mean + EPS)

        rolling_max = df["Close"].shift(1).rolling(w).max()
        df[f"dd_{w}"] = (df["Close"] - rolling_max) / (rolling_max + EPS)

    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    hour = idx.hour.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = idx.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def _make_daily_labels_from_hourly(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    ラベル定義（ベースライン）:
      y_ret  = log(next_close / next_open)
      y_risk = min(0, (next_low - next_open)/next_open)
    """
    df = df_ohlcv.copy()
    df["date"] = df.index.date

    daily = df.groupby("date").agg(
        open_=("Open", "first"),
        close_=("Close", "last"),
        low_=("Low", "min"),
    )

    daily["next_open"] = daily["open_"].shift(-1)
    daily["next_close"] = daily["close_"].shift(-1)
    daily["next_low"] = daily["low_"].shift(-1)

    daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["next_open"] + EPS))
    daily["y_risk"] = np.minimum(
        0.0, (daily["next_low"] - daily["next_open"]) / (daily["next_open"] + EPS)
    )

    return daily[["y_ret", "y_risk"]]


def build_features_for_ticker(
    df_ticker: pd.DataFrame,
    df_market: pd.DataFrame,
    cfg: FeatureConfig,
    ticker: str,
) -> Tuple[pd.DataFrame, List[str]]:
    df = _normalize_index(_ensure_ohlcv(df_ticker), cfg.tz)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[df["High"] >= df["Low"]]

    df = _basic_features(df)
    df = _rolling_features_leak_safe(df, cfg.windows)
    df = _calendar_features(df)

    # market
    mkt = _normalize_index(_ensure_ohlcv(df_market), cfg.tz)
    mkt = mkt.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    mkt = mkt[mkt["High"] >= mkt["Low"]]
    mkt = _basic_features(mkt)
    mkt = _rolling_features_leak_safe(mkt, cfg.windows)
    mkt = mkt.add_prefix("mkt_")

    df = df.merge(mkt, left_index=True, right_index=True, how="inner")

    # relative
    df["rel_ret"] = df["logret_cc"] - df["mkt_logret_cc"]
    df["rel_range"] = df["range"] - df["mkt_range"]
    w = 24 if 24 in cfg.windows else cfg.windows[0]
    df["rel_vol"] = df[f"ret_std_{w}"] / (df[f"mkt_ret_std_{w}"] + EPS)

    if cfg.make_labels:
        daily_labels = _make_daily_labels_from_hourly(df[["Open", "High", "Low", "Close", "Volume"]])
        df["date"] = df.index.date
        df = df.join(daily_labels, on="date")

    df["hour"] = df.index.hour
    df_dec = df[df["hour"] == cfg.decision_hour].copy()
    df_dec["ticker"] = ticker

    label_cols = ["y_ret", "y_risk"] if cfg.make_labels else []
    drop_cols = ["hour", "date"] + label_cols

    feature_cols = [
        c for c in df_dec.columns
        if c not in drop_cols and c != "ticker" and pd.api.types.is_numeric_dtype(df_dec[c])
    ]

    required = feature_cols + label_cols
    df_dec = df_dec.dropna(subset=required)

    return df_dec, feature_cols


# ---------------------------
# IO helpers
# ---------------------------

def load_preprocessed_parquet(prep_root: Path, ticker: str) -> Optional[pd.DataFrame]:
    p = prep_root / ticker / "intraday_1h_preprocessed.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    return df


def save_features_parquet(out_root: Path, ticker: str, df_feat: pd.DataFrame) -> Path:
    out_dir = out_root / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_1h_for_model1.parquet"
    df_feat.to_parquet(out_path, index=True)
    return out_path


# ---------------------------
# main (NO CLI)
# ---------------------------

def main() -> None:
    s = Settings()

    prep_root = s.prep_root.resolve()
    out_root = s.out_root.resolve()

    # TOPIX銘柄取得（.T付き）
    tickers: List[str] = list(extract_topix_codes())

    # 市場コードは確実に確保（特徴生成に必要）
    market_ticker = s.market_ticker
    df_mkt = load_preprocessed_parquet(prep_root, market_ticker)
    if df_mkt is None:
        raise FileNotFoundError(
            f"Market parquet not found: {prep_root / market_ticker / 'intraday_1h_preprocessed.parquet'}"
        )

    # 市場自身は通常「銘柄候補」に入れない
    tickers = [t for t in tickers if t != market_ticker]

    if s.limit_tickers and s.limit_tickers > 0:
        tickers = tickers[: s.limit_tickers]

    cfg = FeatureConfig(
        tz=s.tz,
        windows=s.windows,
        decision_hour=s.decision_hour,
        add_calendar=True,
        add_market=True,
        add_relative=True,
        make_labels=s.make_labels,
    )

    universe_tables: List[pd.DataFrame] = []
    ok = 0
    missing = 0
    failed = 0

    for i, tkr in enumerate(tickers, start=1):
        df = load_preprocessed_parquet(prep_root, tkr)
        if df is None:
            missing += 1
            print(f"[WARN] ({i}/{len(tickers)}) missing parquet: {tkr}")
            continue

        try:
            feat_df, _ = build_features_for_ticker(
                df_ticker=df,
                df_market=df_mkt,
                cfg=cfg,
                ticker=tkr,
            )
            out_path = save_features_parquet(out_root, tkr, feat_df)
            ok += 1
            print(f"[OK] ({i}/{len(tickers)}) {tkr} -> {out_path} (rows={len(feat_df)})")

            if s.save_universe:
                universe_tables.append(feat_df)

        except Exception as e:
            failed += 1
            print(f"[ERROR] ({i}/{len(tickers)}) failed: {tkr} :: {e}")

    print(f"\nDone. ok={ok}, missing={missing}, failed={failed}, requested={len(tickers)}")

    if s.save_universe and universe_tables:
        uni = pd.concat(universe_tables, axis=0, ignore_index=False).sort_index()
        uni_dir = out_root / "universe"
        uni_dir.mkdir(parents=True, exist_ok=True)
        uni_path = uni_dir / "features_1h_for_model1_universe.parquet"
        uni.to_parquet(uni_path, index=True)
        print(f"[OK] universe saved: {uni_path} (rows={len(uni)})")


if __name__ == "__main__":
    main()
