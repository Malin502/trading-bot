import pandas as pd
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

STOCK_PATH = PROJECT_ROOT / "datasets/intraday_1h_ae/stock_latent32.parquet"
MKT_PATH   = PROJECT_ROOT / "datasets/intraday_1h_ae/market_latent32.parquet"
OUT_PATH   = PROJECT_ROOT / "datasets/intraday_1h_ae/latent32_with_mkt.parquet"

KEY = "end_ts"  # 基本これでOK（dateより安全）

stock = pd.read_parquet(STOCK_PATH)
mkt   = pd.read_parquet(MKT_PATH)

# 1) market側は end_ts ごとに1行になってるのが理想。念のため重複排除。
mkt_cols = [c for c in mkt.columns if c.startswith("mkt_z")]
mkt = mkt[[KEY] + mkt_cols].drop_duplicates(subset=[KEY]).reset_index(drop=True)

# 2) 銘柄側に market latent を付与（left join）
out = stock.merge(mkt, on=KEY, how="left")

# 3) 欠損チェック（ズレがあるとここで検出される）
missing_ratio = out[mkt_cols].isna().any(axis=1).mean()
print("missing_ratio:", float(missing_ratio))

# 欠損があるならここで止める（安全）
if missing_ratio > 0:
    bad = out.loc[out[mkt_cols].isna().any(axis=1), [KEY, "ticker"]].head(20)
    raise ValueError(
        f"Market latent missing for some rows. missing_ratio={missing_ratio}\n"
        f"Examples:\n{bad}"
    )

out.to_parquet(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "shape=", out.shape)
print("Example cols:", out.columns[:15].tolist(), "...")