from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# Settings
# ----------------------------

@dataclass
class DatasetSettings:
    universe_parquet: Optional[Path] = Path(PROJECT_ROOT / "features/universe/features_1h_for_model1_universe.parquet")

    # 必須列
    id_col: str = "ticker"
    time_col: str = "Datetime"   # parquetによっては index のことが多い。後で吸収
    y_cols: Tuple[str, str] = ("y_ret", "y_risk")

    # 例: y_ret/y_risk を作っていない推論用特徴の場合は False にする
    require_labels: bool = True

    # データ品質（任意）
    drop_inf: bool = True


# ----------------------------
# IO: load feature tables
# ----------------------------

def _read_parquet_with_datetime_index(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # indexにDatetimeが入ってる想定を吸収
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index.name = "Datetime"
        return df
    # Datetime列がある場合
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
        return df
    raise ValueError(f"Datetime index/column not found in: {path}")


def load_universe_table(s: DatasetSettings) -> pd.DataFrame:
    if s.universe_parquet is not None and s.universe_parquet.exists():
        df = _read_parquet_with_datetime_index(s.universe_parquet)
        return df
    # fallback: scan per ticker
    return load_from_feature_dirs(s)


def load_from_feature_dirs(s: DatasetSettings) -> pd.DataFrame:
    root = s.features_root
    if not root.exists():
        raise FileNotFoundError(f"features_root not found: {root}")

    tables: List[pd.DataFrame] = []
    # ../../features/xxxx.T/features_1h_for_model1.parquet を想定
    for d in sorted(root.glob("*.T")):
        p = d / "features_1h_for_model1.parquet"
        if not p.exists():
            continue
        df = _read_parquet_with_datetime_index(p)
        tables.append(df)

    if not tables:
        raise FileNotFoundError(f"No feature parquet found under: {root}/xxxx.T/")

    out = pd.concat(tables, axis=0, ignore_index=False).sort_index()
    return out


# ----------------------------
# feature cols inference & cleaning
# ----------------------------

def infer_feature_cols(df: pd.DataFrame, id_col: str, y_cols: Sequence[str]) -> List[str]:
    drop = set([id_col, *y_cols])
    # 数値列だけ
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns inferred.")
    return cols


def clean_table(
    df: pd.DataFrame,
    s: DatasetSettings,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()

    if s.drop_inf:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # id列必須
    if s.id_col not in df.columns:
        raise ValueError(f"'{s.id_col}' column not found. columns={list(df.columns)[:20]}...")

    # ラベルが必要なら存在チェック
    if s.require_labels:
        for y in s.y_cols:
            if y not in df.columns:
                raise ValueError(f"label column '{y}' not found. If inference-only, set require_labels=False")

    # feature_cols固定
    if feature_cols is None:
        feature_cols = infer_feature_cols(df, s.id_col, s.y_cols if s.require_labels else [])
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols missing in df: {missing[:10]}...")

    required = feature_cols + ([*s.y_cols] if s.require_labels else [])
    df = df.dropna(subset=required)

    # 時系列順に
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")
    df = df.sort_index()

    return df, feature_cols


# ----------------------------
# Walk-forward split
# ----------------------------

@dataclass
class WalkForwardConfig:
    # 例: train 240営業日, val 20営業日, test 20営業日
    train_days: int = 240
    val_days: int = 20
    test_days: int = 20
    step_days: int = 20  # 次のfoldへ進む幅（通常 test_days と同じでOK）
    min_unique_days: int = 320  # 最低日数

def unique_trading_days(df: pd.DataFrame) -> np.ndarray:
    # DatetimeIndex → 日付でユニーク
    days = pd.Index(df.index.date).unique()
    return np.array(days)

def make_walk_forward_folds(df: pd.DataFrame, cfg: WalkForwardConfig) -> List[Dict[str, np.ndarray]]:
    days = unique_trading_days(df)
    if len(days) < cfg.min_unique_days:
        raise ValueError(f"Not enough unique days: {len(days)} < {cfg.min_unique_days}")

    folds = []
    start = 0
    while True:
        train_end = start + cfg.train_days
        val_end = train_end + cfg.val_days
        test_end = val_end + cfg.test_days
        if test_end > len(days):
            break

        fold = {
            "train_days": days[start:train_end],
            "val_days": days[train_end:val_end],
            "test_days": days[val_end:test_end],
        }
        folds.append(fold)

        start += cfg.step_days

    if not folds:
        raise ValueError("No folds generated. Adjust WalkForwardConfig.")
    return folds

def select_by_days(df: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    mask = np.isin(df.index.date, days)
    return df.loc[mask].copy()


# ----------------------------
# Scaling (leak-safe)
# ----------------------------

@dataclass
class FoldScaler:
    scaler: StandardScaler
    feature_cols: List[str]

    @staticmethod
    def fit(train_df: pd.DataFrame, feature_cols: List[str], exclude_cols: Optional[List[str]] = None) -> "FoldScaler":
        exclude_cols = exclude_cols or []
        cols = [c for c in feature_cols if c not in exclude_cols]
        sc = StandardScaler()
        sc.fit(train_df[cols].to_numpy(dtype=np.float32))
        return FoldScaler(scaler=sc, feature_cols=cols)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        return self.scaler.transform(X).astype(np.float32)


# ----------------------------
# PyTorch Dataset
# ----------------------------

class Model1Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X)  # [N, D]
        self.y = None if y is None else torch.from_numpy(y)  # [N, 2]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ----------------------------
# Public API: build datasets per fold
# ----------------------------

@dataclass
class DataLoadersConfig:
    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = True

def build_fold_datasets_and_loaders(
    df: pd.DataFrame,
    feature_cols: List[str],
    fold: Dict[str, np.ndarray],
    s: DatasetSettings,
    dl_cfg: DataLoadersConfig = DataLoadersConfig(),
) -> Dict[str, object]:
    # split
    train_df = select_by_days(df, fold["train_days"])
    val_df = select_by_days(df, fold["val_days"])
    test_df = select_by_days(df, fold["test_days"])

    # scaler: train only (leak-safe)
    # sin/cosは標準化不要なので除外するのが推奨
    exclude = [c for c in feature_cols if c.endswith("_sin") or c.endswith("_cos")]
    fs = FoldScaler.fit(train_df, feature_cols, exclude_cols=exclude)

    X_train = fs.transform(train_df)
    X_val = fs.transform(val_df)
    X_test = fs.transform(test_df)

    if s.require_labels:
        y_train = train_df[list(s.y_cols)].to_numpy(dtype=np.float32)
        y_val = val_df[list(s.y_cols)].to_numpy(dtype=np.float32)
        y_test = test_df[list(s.y_cols)].to_numpy(dtype=np.float32)
    else:
        y_train = y_val = y_test = None

    ds_train = Model1Dataset(X_train, y_train)
    ds_val = Model1Dataset(X_val, y_val)
    ds_test = Model1Dataset(X_test, y_test)

    # shuffleは train だけ（時系列リークは起きない。日付でsplit済みのため）
    dl_train = DataLoader(ds_train, batch_size=dl_cfg.batch_size, shuffle=True,
                          num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=dl_cfg.batch_size, shuffle=False,
                        num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=dl_cfg.batch_size, shuffle=False,
                         num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory, drop_last=False)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols_scaled": fs.feature_cols,  # 実際にスケールした列
        "scaler": fs.scaler,
        "dl_train": dl_train,
        "dl_val": dl_val,
        "dl_test": dl_test,
    }


# ----------------------------
# main-like entry (no CLI)
# ----------------------------

def main_build_datasets() -> None:
    s = DatasetSettings(
        universe_parquet=Path(PROJECT_ROOT / "features/universe/features_1h_for_model1_universe.parquet"),
        require_labels=True,  # 学習ならTrue
    )

    df = load_universe_table(s)

    # feature cols固定（この順序を保存しておくのが重要）
    df, feature_cols = clean_table(df, s, feature_cols=None)

    wf = WalkForwardConfig(
        train_days=240,
        val_days=20,
        test_days=20,
        step_days=20,
        min_unique_days=320,
    )
    folds = make_walk_forward_folds(df, wf)
    print(f"folds={len(folds)}, total_rows={len(df)}, unique_days={len(unique_trading_days(df))}")
    print(f"n_features={len(feature_cols)}")

    # 例: 0番foldを作ってみる
    dl_cfg = DataLoadersConfig(batch_size=512, num_workers=0)
    pack = build_fold_datasets_and_loaders(df, feature_cols, folds[0], s, dl_cfg)

    print("train/val/test rows:",
          len(pack["train_df"]), len(pack["val_df"]), len(pack["test_df"]))
    print("scaled feature cols:", len(pack["feature_cols_scaled"]))


if __name__ == "__main__":
    main_build_datasets()
