import pandas as pd

# ファイルを読み込む
df = pd.read_parquet('data/clean/1925.T/intraday_15m_2025-02.parquet')

# 先頭を表示
print(df.tail())

# 情報を表示
print(df.info())

# 統計情報
print(df.describe())