from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class FoldIndex:
    fold_id: int
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]

def make_walkforward_folds(
    df: pd.DataFrame,
    train_days: int = 240,
    val_days: int = 20,
    test_days: int = 20,
    step_days: int = 20,
    purge_days: int = 1,
) -> list[FoldIndex]:
    """
    df: 1行=1銘柄×1時点(15:00) を含む想定。datetimeで日付単位に折る。
    """
    # ユニーク日付（datetime→date）
    dates = pd.to_datetime(df["datetime"]).dt.date
    uniq_days = pd.Index(sorted(dates.unique()))
    n = len(uniq_days)

    folds: list[FoldIndex] = []
    start = 0
    fold_id = 0
    total_window = train_days + val_days + test_days

    while start + total_window <= n:
        train_days_range = uniq_days[start : start + train_days]
        val_start = start + train_days + purge_days
        val_end = val_start + val_days
        test_start = val_end + purge_days
        test_end = test_start + test_days

        if test_end > n:
            break

        val_days_range   = uniq_days[val_start : val_end]
        test_days_range  = uniq_days[test_start : test_end]

        day_series = dates  # already date
        train_mask = day_series.isin(train_days_range)
        val_mask   = day_series.isin(val_days_range)
        test_mask  = day_series.isin(test_days_range)

        train_idx = df.index[train_mask].tolist()
        val_idx   = df.index[val_mask].tolist()
        test_idx  = df.index[test_mask].tolist()

        folds.append(FoldIndex(fold_id, train_idx, val_idx, test_idx))
        fold_id += 1
        start += step_days

    return folds
