# datasets_1h.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ROOT = PROJECT_ROOT / "features"


@dataclass(frozen=True)
class Sample:
    ticker: str
    start_idx: int
    end_idx: int
    y: int


class WindowedFeaturesDataset(Dataset):
    """
    Returns:
      X: (N, D) float32
      y: int64 (0/1)
    """
    def __init__(self, manifest_parquet: str | Path):
        self.manifest = pd.read_parquet(manifest_parquet)
        # Ensure columns exist
        for col in ["ticker", "start_idx", "end_idx", "y"]:
            if col not in self.manifest.columns:
                raise ValueError(f"manifest missing column: {col}")

        self.samples = [
            Sample(
                ticker=row["ticker"],
                start_idx=int(row["start_idx"]),
                end_idx=int(row["end_idx"]),
                y=int(row["y"]),
            )
            for _, row in self.manifest.iterrows()
        ]

        # cache per ticker
        self._feat_cache: Dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_features(self, ticker: str) -> pd.DataFrame:
        if ticker in self._feat_cache:
            return self._feat_cache[ticker]
        fp = FEAT_ROOT / ticker / "features_1h_for_ae.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Missing features: {fp}")
        df = pd.read_parquet(fp)
        if "Datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("Datetime")
        df = df.sort_index()
        self._feat_cache[ticker] = df
        return df

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        feat = self._load_features(s.ticker)
        window = feat.iloc[s.start_idx : s.end_idx + 1]

        x = window.to_numpy(dtype=np.float32)  # (N, D)
        y = np.int64(s.y)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.int64)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    dataset = WindowedFeaturesDataset(
        manifest_parquet=PROJECT_ROOT / "datasets" / "intraday_1h_ae" / "fold_000_train.parquet"
    )
    print(f"Dataset size: {len(dataset)}")

    # Example: get first sample
    x, y = dataset[0]
    print(f"x shape: {x.shape}, y: {y}")