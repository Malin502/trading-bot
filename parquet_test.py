import pandas as pd
import pathlib

# ファイルを読み込む
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
df = pd.read_parquet(PROJECT_ROOT / 'features/1925.T/features_1h_for_model1.parquet')

# 先頭を表示
print(df.tail())

# 情報を表示
print(df.info())

# 統計情報
print(df.describe())

print(df.columns.tolist())