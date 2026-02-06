import pandas as pd

# ファイルを読み込む
df = pd.read_parquet('data/preprocessing/1306.T/intraday_1h_preprocessed.parquet')

# 先頭を表示
print(df.tail())

# 情報を表示
print(df.info())

# 統計情報
print(df.describe())

print(df.columns.tolist())