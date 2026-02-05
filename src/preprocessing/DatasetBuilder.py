# build_dataset_manifest_1h.py
from __future__ import annotations
import sys

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.market_data.CodeExtractor import extract_topix_codes

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ROOT = PROJECT_ROOT / "features"
PREP_ROOT = PROJECT_ROOT / "data" / "preprocessing"
OUT_ROOT = PROJECT_ROOT / "datasets" / "intraday_1h_ae"

MARKET_TICKER = "1306.T"

N_BARS = 56 # 7 BARS_PER_DAY × 8 days
BARS_PER_DAY = 7
STRIDE_DAYS = 1

# walk-forward fold parameters
TRAIN_DAYS = 252
VAL_DAYS = 63
TEST_DAYS = 63
STEP_DAYS = TEST_DAYS  # 次のfoldへ進む幅


def _read_features(ticker: str) -> pd.DataFrame:
    fp = FEAT_ROOT / ticker / "features_1h_for_ae.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing features: {fp}")
    df = pd.read_parquet(fp)
    if "Datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("Datetime")
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: features index is not DatetimeIndex")
    return df


def _read_ohlcv(ticker: str) -> pd.DataFrame:
    fp = PREP_ROOT / ticker / "intraday_1h_preprocessed.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing OHLCV: {fp}")
    df = pd.read_parquet(fp)
    if "Datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("Datetime")
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: ohlcv index is not DatetimeIndex")
    return df


def _daily_last_timestamps(index: pd.DatetimeIndex) -> pd.Series:
    """Return Series: date -> last timestamp in that date."""
    s = pd.Series(index, index=index)
    dates = pd.Series(index.date, index=index)
    # last timestamp per date
    last_ts = s.groupby(dates).max()
    last_ts.index = pd.Index(last_ts.index, name="date")
    last_ts.name = "end_ts"
    return last_ts


def _build_samples_for_ticker(ticker: str) -> pd.DataFrame:
    feat = _read_features(ticker)
    ohlcv = _read_ohlcv(ticker)

    # Make sure we can compute label based on Close at daily last bar
    last_feat = _daily_last_timestamps(feat.index)
    last_px = _daily_last_timestamps(ohlcv.index)

    # Use intersection of available dates
    common_dates = last_feat.index.intersection(last_px.index)
    last_feat = last_feat.loc[common_dates]
    last_px = last_px.loc[common_dates]

    # Sort by date
    last_feat = last_feat.sort_index()
    last_px = last_px.sort_index()

    # Build mapping: end_ts -> integer position in features (for slicing fast)
    feat_index = feat.index
    pos_map = pd.Series(np.arange(len(feat_index)), index=feat_index)

    rows = []
    dates = list(last_feat.index)

    # stride: 1 day
    for i in range(len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]

        end_ts = last_feat.loc[d]
        end_ts_next = last_px.loc[d_next]  # label uses next day close

        # Need end_ts exist in features index (it should)
        if end_ts not in pos_map.index:
            continue
        end_idx = int(pos_map.loc[end_ts])

        start_idx = end_idx - (N_BARS - 1)
        if start_idx < 0:
            continue

        # Ensure the window is contiguous in bar count (no missing bars inside)
        window = feat.iloc[start_idx : end_idx + 1]
        if len(window) != N_BARS:
            continue

        # Drop samples containing any NaN (決定済み)
        if window.isna().any().any():
            continue

        # Label: next day Close (daily last bar) - today Close (daily last bar) > 0
        # Today close at its daily last bar
        end_ts_px = last_px.loc[d]
        close_today = float(ohlcv.loc[end_ts_px, "Close"]) if end_ts_px in ohlcv.index else np.nan
        close_next = float(ohlcv.loc[end_ts_next, "Close"]) if end_ts_next in ohlcv.index else np.nan
        if not np.isfinite(close_today) or not np.isfinite(close_next):
            continue

        y = 1 if (close_next - close_today) > 0 else 0

        rows.append(
            {
                "ticker": ticker,
                "date": pd.Timestamp(d),
                "end_ts": end_ts,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "y": y,
            }
        )

    return pd.DataFrame(rows)


def _make_walk_forward_folds(all_dates: List[pd.Timestamp]) -> List[Dict[str, str]]:
    """
    Build folds based on a global date list.
    Each fold returns date ranges (inclusive start, inclusive end) for train/val/test.
    """
    folds = []
    total = len(all_dates)
    cursor = 0

    while True:
        train_start = cursor
        train_end = train_start + TRAIN_DAYS - 1
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        if test_end >= total:
            remaining = total - cursor
            if remaining > 0:
                print(f"Warning: Insufficient data for complete fold. Remaining days: {remaining}, Required: {TRAIN_DAYS + VAL_DAYS + TEST_DAYS}")
            break

        fold = {
            "train_start": str(all_dates[train_start].date()),
            "train_end": str(all_dates[train_end].date()),
            "val_start": str(all_dates[train_end + 1].date()),
            "val_end": str(all_dates[val_end].date()),
            "test_start": str(all_dates[val_end + 1].date()),
            "test_end": str(all_dates[test_end].date()),
        }
        folds.append(fold)
        cursor += STEP_DAYS

    return folds


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    tickers = list(extract_topix_codes())[:100]

    # Build samples for all tickers
    all_samples = []
    errors = []

    for i, tkr in enumerate(tickers, 1):
        try:
            df = _build_samples_for_ticker(tkr)
            all_samples.append(df)
            print(f"[{i:03d}/{len(tickers)}] OK {tkr} samples={len(df)}")
        except Exception as e:
            errors.append({"ticker": tkr, "error": repr(e)})
            print(f"[{i:03d}/{len(tickers)}] ERR {tkr} {repr(e)}")

    samples = pd.concat(all_samples, axis=0, ignore_index=True) if all_samples else pd.DataFrame()
    
    if samples.empty:
        print("Error: No samples were created. Check data files and preprocessing steps.")
        if errors:
            (OUT_ROOT / "_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved errors: {OUT_ROOT / '_errors.json'} count={len(errors)}")
        return
    
    samples = samples.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Save master samples
    samples_fp = OUT_ROOT / "samples.parquet"
    samples.to_parquet(samples_fp)
    print(f"Saved: {samples_fp} rows={len(samples)}")

    # Build global date axis from market ticker (stable reference)
    # Use OHLCV daily-last dates for market ticker
    try:
        mkt_ohlcv = _read_ohlcv(MARKET_TICKER)
    except Exception as e:
        print(f"Error: Failed to load market ticker {MARKET_TICKER}: {repr(e)}")
        print("Cannot build folds without market reference data.")
        if errors:
            (OUT_ROOT / "_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved errors: {OUT_ROOT / '_errors.json'} count={len(errors)}")
        return
    
    mkt_last = _daily_last_timestamps(mkt_ohlcv.index).sort_index()
    all_dates = [pd.Timestamp(d) for d in mkt_last.index]

    folds = _make_walk_forward_folds(all_dates)
    folds_fp = OUT_ROOT / "folds.json"
    folds_fp.write_text(json.dumps({"folds": folds, "params": {
        "TRAIN_DAYS": TRAIN_DAYS, "VAL_DAYS": VAL_DAYS, "TEST_DAYS": TEST_DAYS,
        "STEP_DAYS": STEP_DAYS, "N_BARS": N_BARS, "BARS_PER_DAY": BARS_PER_DAY
    }}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {folds_fp} folds={len(folds)}")

    # Materialize per-fold sample lists
    for k, f in enumerate(folds):
        tr_start = pd.Timestamp(f["train_start"])
        tr_end = pd.Timestamp(f["train_end"])
        va_start = pd.Timestamp(f["val_start"])
        va_end = pd.Timestamp(f["val_end"])
        te_start = pd.Timestamp(f["test_start"])
        te_end = pd.Timestamp(f["test_end"])

        train_df = samples[(samples["date"] >= tr_start) & (samples["date"] <= tr_end)]
        val_df = samples[(samples["date"] >= va_start) & (samples["date"] <= va_end)]
        test_df = samples[(samples["date"] >= te_start) & (samples["date"] <= te_end)]

        train_df.to_parquet(OUT_ROOT / f"fold_{k:03d}_train.parquet")
        val_df.to_parquet(OUT_ROOT / f"fold_{k:03d}_val.parquet")
        test_df.to_parquet(OUT_ROOT / f"fold_{k:03d}_test.parquet")

        print(f"[fold {k:03d}] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Save errors if any
    if errors:
        (OUT_ROOT / "_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved errors: {OUT_ROOT / '_errors.json'} count={len(errors)}")


if __name__ == "__main__":
    main()
