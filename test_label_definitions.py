"""ラベル定義のテスト"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.FeatureBuilder1h import _make_daily_labels_from_hourly

# サンプルデータ作成
dates = pd.date_range("2024-01-01 09:00", periods=30, freq="1H")
df = pd.DataFrame({
    "Open": 100 + np.random.randn(30) * 2,
    "High": 102 + np.random.randn(30) * 2,
    "Low": 98 + np.random.randn(30) * 2,
    "Close": 100 + np.random.randn(30) * 2,
    "Volume": 1000000 + np.random.randn(30) * 100000,
}, index=dates)

# 各ラベルタイプをテスト
for label_type in ["oc", "cc", "oo"]:
    print(f"\n=== label_type: {label_type} ===")
    labels = _make_daily_labels_from_hourly(df, label_type=label_type)
    print(labels.head(10))
    print(f"y_ret stats: mean={labels['y_ret'].mean():.6f}, std={labels['y_ret'].std():.6f}")
    print(f"y_risk stats: mean={labels['y_risk'].mean():.6f}, std={labels['y_risk'].std():.6f}")

print("\nラベル定義テスト完了")
