import pandas as pd

# ファイルを読み込む
df = pd.read_parquet('features/1925.T/features_1h_for_ae.parquet')

# 先頭を表示
print(df.tail())

# 情報を表示
print(df.info())

# 統計情報
print(df.describe())