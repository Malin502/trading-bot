from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import pandas as pd
import numpy as np

REQUIRED_COLS = ["datetime", "ticker", "y_ret", "y_risk"]

def load_universe_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f"missing required cols: {missing}"

    # datetimeは必ずdatetime64へ
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)

    # 安全: ソート
    df = df.sort_values(["datetime", "ticker"]).reset_index(drop=True)

    return df

def infer_feature_cols(df: pd.DataFrame, feature_cols: Sequence[str] | None) -> list[str]:
    if feature_cols is not None:
        for c in feature_cols:
            assert c in df.columns, f"feature col not found: {c}"
        return list(feature_cols)

    # 自動推論: required以外で、数値列のみを採用
    drop = set(REQUIRED_COLS)
    cand = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cand.append(c)
    assert len(cand) > 0, "no numeric feature cols inferred"
    return cand

def df_to_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_ret = df["y_ret"].to_numpy(dtype=np.float32)
    y_risk = df["y_risk"].to_numpy(dtype=np.float32)
    return X, y_ret, y_risk
