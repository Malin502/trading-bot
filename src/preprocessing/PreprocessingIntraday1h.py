# preprocess_intraday_1h.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.market_data.CodeExtractor import extract_topix_codes

root_path = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = root_path / "data/clean"
OUT_ROOT = root_path / "data/preprocessing"


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _ensure_datetime_index_jst(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # If tz-naive, localize to JST; if tz-aware, convert to JST
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Tokyo")
    else:
        df.index = df.index.tz_convert("Asia/Tokyo")

    return df


def _load_raw_intraday_1h(ticker: str) -> pd.DataFrame:
    ticker_dir = RAW_ROOT / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {ticker_dir}")

    files = sorted(ticker_dir.glob("intraday_1h_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {ticker_dir}/intraday_1h_*.parquet")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        # Some pipelines store datetime as a column; try to set index if needed
        if "Datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("Datetime")
        dfs.append(df)

    df_all = pd.concat(dfs, axis=0, copy=False)
    df_all = _ensure_datetime_index_jst(df_all)

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df_all.columns]
    if missing:
        raise ValueError(f"{ticker}: missing required columns: {missing}")

    # Sort & drop duplicate timestamps
    df_all = df_all.sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]

    # Keep only required columns (order fixed)
    df_all = df_all[REQUIRED_COLS]

    return df_all


def _infer_valid_times_top7(df: pd.DataFrame) -> List[str]:
    """
    Infer the 7 most common HH:MM times in the index.
    Returns list like ["09:00", "10:00", ...]
    """
    times = pd.Series(df.index.strftime("%H:%M"))
    counts = times.value_counts()
    top7 = counts.head(7).index.tolist()
    if len(top7) < 7:
        raise ValueError(f"Found only {len(top7)} unique times; cannot enforce 7 bars/day.")
    return sorted(top7)


def _filter_valid_times(df: pd.DataFrame, valid_times: Iterable[str]) -> pd.DataFrame:
    vt = set(valid_times)
    mask = df.index.strftime("%H:%M").isin(vt)
    return df.loc[mask]


def _drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Drop NaNs
    df = df.dropna(subset=REQUIRED_COLS)

    # Drop non-positive OHLC (0 or negative is invalid)
    for c in ["Open", "High", "Low", "Close"]:
        df = df[df[c] > 0]

    # Drop volume <= 0 (if you want to keep volume==0, change to df[df["Volume"] >= 0])
    df = df[df["Volume"] >= 0]

    # Optional sanity: High >= max(Open,Close), Low <= min(Open,Close)
    df = df[df["High"] >= df[["Open", "Close"]].max(axis=1)]
    df = df[df["Low"] <= df[["Open", "Close"]].min(axis=1)]

    return df


def _keep_full_days_7bars(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Keep only dates that have exactly 7 bars.
    Returns (filtered_df, kept_day_count)
    """
    dates = pd.Series(df.index.date, index=df.index)
    counts = dates.value_counts()
    full_days = set(counts[counts == 7].index)

    mask = dates.isin(full_days)
    df2 = df.loc[mask].copy()
    kept_days = len(full_days)

    # After filtering, keep order
    df2 = df2.sort_index()
    return df2, kept_days


def preprocess_one(ticker: str) -> dict:
    df = _load_raw_intraday_1h(ticker)

    # Infer valid 7 times from this ticker's data
    valid_times = _infer_valid_times_top7(df)

    # Apply time filter
    df = _filter_valid_times(df, valid_times)

    # Drop bad rows
    df = _drop_bad_rows(df)

    # Keep only full days with 7 bars
    before_rows = len(df)
    df, kept_days = _keep_full_days_7bars(df)
    after_rows = len(df)

    # Output paths
    out_dir = OUT_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / "intraday_1h_preprocessed.parquet"
    df.to_parquet(out_parquet)

    meta = {
        "ticker": ticker,
        "valid_times_hhmm": valid_times,
        "rows_before_full_day_filter": int(before_rows),
        "rows_after_full_day_filter": int(after_rows),
        "kept_full_days": int(kept_days),
        "time_min": str(df.index.min()) if len(df) else None,
        "time_max": str(df.index.max()) if len(df) else None,
        "columns": REQUIRED_COLS,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    codes = extract_topix_codes()  # returns list with ".T"
    codes = list(codes)[:100]      # ensure 100 tickers
    
    codes = ['1306.T']  # For testing, process only TOPIX ETF

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for i, tkr in enumerate(codes, 1):
        try:
            meta = preprocess_one(tkr)
            results.append(meta)
            print(f"[{i:03d}/{len(codes)}] OK  {tkr}  days={meta['kept_full_days']}  times={meta['valid_times_hhmm']}")
        except Exception as e:
            errors.append({"ticker": tkr, "error": repr(e)})
            print(f"[{i:03d}/{len(codes)}] ERR {tkr}  {repr(e)}")

    # Save run summary
    summary = {
        "processed": len(results),
        "failed": len(errors),
        "errors": errors,
    }
    (OUT_ROOT / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. processed={len(results)} failed={len(errors)}")
    print(f"Summary: {OUT_ROOT / '_summary.json'}")


if __name__ == "__main__":
    main()
