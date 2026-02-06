import pathlib
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

df = pd.read_parquet(PROJECT_ROOT / "datasets/intraday_1h_ae/latent32.parquet")
print((df["ticker"]=="1306.T").sum())