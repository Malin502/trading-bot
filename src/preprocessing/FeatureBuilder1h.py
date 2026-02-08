from __future__ import annotations

import json
import sys
import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.market_data.CodeExtractor import extract_topix_codes


EPS = 1e-8


# ---------------------------
# Settings (EDIT HERE)
# ---------------------------

@dataclass
class Settings:
    # 入力（前処理済み）データのルート
    prep_root: Path = Path(PROJECT_ROOT / "data/preprocessing")

    # 出力（特徴量）データのルート
    out_root: Path = Path(PROJECT_ROOT / "features")

    # 市場コード（同期インデックス）
    market_ticker: str = "1306.T"

    # セクターマップ（ticker -> sector）
    sector_map_path: Path = Path(PROJECT_ROOT / "config/sector_mapping.json")
    use_sector_features: bool = True
    
    # ラベルタイプ: "oc" (寄→引), "cc" (引→引), "oo" (寄→寄)
    label_type: str = os.environ.get("LABEL_TYPE", "oc")

    # 何銘柄処理するか（0=全TOPIX、例: 50なら先頭50銘柄）
    limit_tickers: int = 0

    # 全銘柄を結合したuniverse parquetも保存するか
    save_universe: bool = True

    # ラベル(y_ret,y_risk)を作成するか（学習用途ならTrue）
    make_labels: bool = True
    
    # ラベルタイプ: "oc" (寄→引), "cc" (引→引), "oo" (寄→寄)
    label_type: str = "oc"

    # 15:00バーを「その日の決定時刻」として切り出す
    decision_hour: int = 15

    # rolling window（1時間足）
    windows: Tuple[int, ...] = (8, 24, 56)

    # timezone
    tz: str = "Asia/Tokyo"


# ---------------------------
# Feature Builder (Model①)
# ---------------------------

@dataclass
class FeatureConfig:
    tz: str
    windows: Tuple[int, ...]
    decision_hour: int
    add_calendar: bool = True
    add_market: bool = True
    add_relative: bool = True
    make_labels: bool = True
    label_type: str = "oc"


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.copy()


def _normalize_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy().sort_index()
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    df.index = idx
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    df["logret_cc"] = np.log((c + EPS) / (c.shift(1) + EPS))
    df["logret_oc"] = np.log((c + EPS) / (o + EPS))

    df["range"] = (h - l) / (c + EPS)
    df["body"] = (c - o) / (c + EPS)
    df["upper_wick"] = (h - np.maximum(o, c)) / (c + EPS)
    df["lower_wick"] = (np.minimum(o, c) - l) / (c + EPS)

    df["v_log"] = np.log(v + 1.0)
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


def _rolling_features_leak_safe(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
    """
    IMPORTANT: shift(1) -> rolling()
    """
    df = df.copy()
    for w in windows:
        df[f"ret_std_{w}"] = df["logret_cc"].shift(1).rolling(w).std()
        df[f"ret_mean_{w}"] = df["logret_cc"].shift(1).rolling(w).mean()

        df[f"range_mean_{w}"] = df["range"].shift(1).rolling(w).mean()
        df[f"range_std_{w}"] = df["range"].shift(1).rolling(w).std()

        vol_mean = df["Volume"].shift(1).rolling(w).mean()
        df[f"v_rel_{w}"] = df["Volume"] / (vol_mean + EPS)

        rolling_max = df["Close"].shift(1).rolling(w).max()
        df[f"dd_{w}"] = (df["Close"] - rolling_max) / (rolling_max + EPS)

    return df


def add_technical_features(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
    df = df.copy()

    c = df["Close"].shift(1)
    h = df["High"].shift(1)
    l = df["Low"].shift(1)
    v = df["Volume"].shift(1)

    for w in windows:
        df[f"ret_{w}"] = np.log((c + EPS) / (c.shift(w) + EPS))

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / w, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1.0 / w, adjust=False, min_periods=1).mean()
        rs = avg_gain / (avg_loss + EPS)
        df[f"rsi_{w}"] = 100.0 - (100.0 / (1.0 + rs))

        # Bollinger
        ma = c.rolling(w, min_periods=3).mean()
        sd = c.rolling(w, min_periods=3).std()
        df[f"bb_z_{w}"] = (c - ma) / (sd + EPS)
        df[f"bb_width_{w}"] = (2.0 * sd) / (ma + EPS)

        # ATR
        prev_close = c.shift(1)
        tr = pd.concat(
            [(h - l), (h - prev_close).abs(), (l - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        df[f"atr_{w}"] = tr.rolling(w, min_periods=3).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / (c + EPS)

        # ADX
        up_move = h.diff()
        down_move = -l.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )
        atr_w = tr.rolling(w, min_periods=3).mean()
        plus_di = 100.0 * plus_dm.rolling(w, min_periods=3).sum() / (atr_w + EPS)
        minus_di = 100.0 * minus_dm.rolling(w, min_periods=3).sum() / (atr_w + EPS)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
        df[f"adx_{w}"] = dx.rolling(w, min_periods=3).mean()

    fast = 8 if 8 in windows else max(4, windows[0] // 2)
    slow = 24 if 24 in windows else max(windows)
    signal = 9
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    df["macd_line"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    # OBV
    direction = np.sign(c.diff().fillna(0.0))
    df["obv"] = (direction * v).fillna(0.0).cumsum()
    for w in windows:
        obv_ma = df["obv"].shift(1).rolling(w, min_periods=3).mean()
        obv_sd = df["obv"].shift(1).rolling(w, min_periods=3).std()
        df[f"obv_z_{w}"] = (df["obv"] - obv_ma) / (obv_sd + EPS)

    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    hour = idx.hour.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = idx.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def _load_sector_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"sector_map must be a dict: {path}")
    return {str(k): str(v) for k, v in data.items()}


def add_market_regime_features(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        mkt_ma = df["mkt_Close"].shift(1).rolling(w, min_periods=1).mean()
        df[f"mkt_trend_{w}"] = (df["mkt_Close"] - mkt_ma) / (mkt_ma + EPS)

        if f"mkt_ret_std_{w}" in df.columns:
            long_w = w * 3
            mkt_vol_long = df[f"mkt_ret_std_{w}"].shift(1).rolling(long_w, min_periods=1).mean()
            df[f"mkt_vol_regime_{w}"] = df[f"mkt_ret_std_{w}"] / (mkt_vol_long + EPS)

    if "mkt_logret_cc" in df.columns:
        ret_sign = np.sign(df["mkt_logret_cc"])
        df["mkt_momentum_consistency"] = ret_sign.shift(1).rolling(5, min_periods=1).mean().abs()

    return df


def add_liquidity_features(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
    df = df.copy()

    for w in windows:
        df[f"volume_abs_mean_{w}"] = np.log(
            df["Volume"].shift(1).rolling(w, min_periods=1).mean() + 1.0
        )

        vol_mean = df["Volume"].shift(1).rolling(w, min_periods=3).mean()
        vol_std = df["Volume"].shift(1).rolling(w, min_periods=3).std()
        df[f"volume_z_{w}"] = (df["Volume"].shift(1) - vol_mean) / (vol_std + EPS)
        df[f"volume_mom_{w}"] = df["Volume"].shift(1) / (vol_mean + EPS)

    w_mid = 24 if 24 in windows else windows[0]
    vol_mean = df["Volume"].shift(1).rolling(w_mid, min_periods=1).mean()
    vol_std = df["Volume"].shift(1).rolling(w_mid, min_periods=1).std()
    cv = vol_std / (vol_mean + EPS)
    df[f"volume_stability_{w_mid}"] = 1.0 / (cv + 0.1)

    df[f"spread_proxy_{w_mid}"] = (
        ((df["High"] - df["Low"]) / (df["Close"] + EPS))
        .shift(1)
        .rolling(w_mid, min_periods=1)
        .mean()
    )

    return df


def add_sector_features(
    df: pd.DataFrame,
    df_sector_peers: pd.DataFrame,
    windows: Tuple[int, ...],
) -> pd.DataFrame:
    df = df.copy()
    dt_col = "__dt__"

    left = df.reset_index()
    left_dt_name = left.columns[0]
    left[dt_col] = pd.to_datetime(left[left_dt_name])

    right = df_sector_peers.copy()
    if dt_col not in right.columns:
        if isinstance(right.index, pd.DatetimeIndex):
            right = right.reset_index()
            right[dt_col] = pd.to_datetime(right[right.columns[0]])
        else:
            raise ValueError("df_sector_peers must contain '__dt__' or DatetimeIndex")

    required = [dt_col, "ticker", "sector_ret", "sector_vol"]
    missing = [c for c in required if c not in right.columns]
    if missing:
        raise ValueError(f"df_sector_peers missing columns: {missing}")

    right = right[required].drop_duplicates(subset=[dt_col, "ticker"])

    left = left.merge(
        right,
        on=[dt_col, "ticker"],
        how="left",
        validate="many_to_one",
    )
    left = left.drop(columns=[dt_col])
    df = left.set_index(left_dt_name)
    df.index.name = left_dt_name

    if "logret_cc" in df.columns and "sector_ret" in df.columns:
        df["sector_rel_ret"] = df["logret_cc"] - df["sector_ret"]

        ret = df["logret_cc"].shift(1)
        sec = df["sector_ret"].shift(1)

        for w in windows:
            cov = ret.rolling(w, min_periods=3).cov(sec)
            var = sec.rolling(w, min_periods=3).var()
            beta = cov / (var + EPS)
            df[f"sector_beta_{w}"] = beta.replace([np.inf, -np.inf], np.nan)

    if "sector_ret" in df.columns:
        for w in windows:
            df[f"sector_momentum_{w}"] = (
                df["sector_ret"].shift(1).rolling(w, min_periods=3).mean()
            )
            df[f"sector_momentum_{w}"] = df[f"sector_momentum_{w}"].replace([np.inf, -np.inf], np.nan)

    return df


def build_sector_proxy_data(
    df_universe: pd.DataFrame,
    sector_map: dict[str, str],
    windows: Tuple[int, ...],
) -> pd.DataFrame:
    df = df_universe.copy()
    dt_col = "__dt__"
    df[dt_col] = pd.to_datetime(df.index)
    df["sector"] = df["ticker"].map(sector_map)

    w_ref = 24 if 24 in windows else windows[0]
    ret_std_col = f"ret_std_{w_ref}"
    if ret_std_col not in df.columns:
        raise ValueError(f"{ret_std_col} not found for sector aggregation")

    sector_df = df.dropna(subset=["sector"])
    sector_agg = (
        sector_df.groupby([dt_col, "sector"], as_index=False)
        .agg(
            sector_ret=("logret_cc", "mean"),
            sector_vol=(ret_std_col, "mean"),
        )
    )

    out = (
        df[[dt_col, "ticker", "sector"]]
        .drop_duplicates(subset=[dt_col, "ticker"])
        .merge(sector_agg, on=[dt_col, "sector"], how="left", validate="many_to_one")
        .drop(columns=["sector"])
    )
    out[["sector_ret", "sector_vol"]] = out[["sector_ret", "sector_vol"]].replace([np.inf, -np.inf], np.nan)
    return out[[dt_col, "ticker", "sector_ret", "sector_vol"]]


def _make_daily_labels_from_hourly(df_ohlcv: pd.DataFrame, label_type: str = "oc") -> pd.DataFrame:
    """
    ラベル定義（複数候補）:
      label_type="oc": 寄→引
        y_ret  = log(next_close / next_open)
        y_risk = max(0, (next_open - next_low)/next_open)
      label_type="cc": 引→引
        y_ret  = log(next_close / current_close)
        y_risk = max(0, (current_close - next_low)/current_close)
      label_type="oo": 寄→寄
        y_ret  = log(next_open / current_open)
        y_risk = max(0, (current_open - next_low)/current_open)
    """
    df = df_ohlcv.copy()
    df["date"] = df.index.date

    daily = df.groupby("date").agg(
        open_=("Open", "first"),
        close_=("Close", "last"),
        low_=("Low", "min"),
        high_=("High", "max"),
    )

    daily["next_open"] = daily["open_"].shift(-1)
    daily["next_close"] = daily["close_"].shift(-1)
    daily["next_low"] = daily["low_"].shift(-1)

    if label_type == "oc":
        # 翌日寄→引
        daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["next_open"] + EPS))
        daily["y_risk"] = np.maximum(
            0.0, (daily["next_open"] - daily["next_low"]) / (daily["next_open"] + EPS)
        )
    elif label_type == "cc":
        # 引→翌日引
        daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["close_"] + EPS))
        daily["y_risk"] = np.maximum(
            0.0, (daily["close_"] - daily["next_low"]) / (daily["close_"] + EPS)
        )
    elif label_type == "oo":
        # 寄→翌日寄
        daily["y_ret"] = np.log((daily["next_open"] + EPS) / (daily["open_"] + EPS))
        daily["y_risk"] = np.maximum(
            0.0, (daily["open_"] - daily["next_low"]) / (daily["open_"] + EPS)
        )
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    return daily[["y_ret", "y_risk"]]


def build_features_for_ticker(
    df_ticker: pd.DataFrame,
    df_market: pd.DataFrame,
    cfg: FeatureConfig,
    ticker: str,
) -> Tuple[pd.DataFrame, List[str]]:
    df = _normalize_index(_ensure_ohlcv(df_ticker), cfg.tz)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[df["High"] >= df["Low"]]

    df = _basic_features(df)
    df = _rolling_features_leak_safe(df, cfg.windows)
    df = _calendar_features(df)
    df = add_technical_features(df, cfg.windows)

    # market
    mkt = _normalize_index(_ensure_ohlcv(df_market), cfg.tz)
    mkt = mkt.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    mkt = mkt[mkt["High"] >= mkt["Low"]]
    mkt = _basic_features(mkt)
    mkt = _rolling_features_leak_safe(mkt, cfg.windows)
    mkt = mkt.add_prefix("mkt_")

    df = df.merge(mkt, left_index=True, right_index=True, how="inner")

    # relative
    df["rel_ret"] = df["logret_cc"] - df["mkt_logret_cc"]
    df["rel_range"] = df["range"] - df["mkt_range"]
    w = 24 if 24 in cfg.windows else cfg.windows[0]
    df["rel_vol"] = df[f"ret_std_{w}"] / (df[f"mkt_ret_std_{w}"] + EPS)

    df = add_market_regime_features(df, cfg.windows)
    df = add_liquidity_features(df, cfg.windows)

    if cfg.make_labels:
        daily_labels = _make_daily_labels_from_hourly(
            df[["Open", "High", "Low", "Close", "Volume"]], 
            label_type=cfg.label_type
        )
        df["date"] = df.index.date
        df = df.join(daily_labels, on="date")

    df["hour"] = df.index.hour
    df_dec = df[df["hour"] == cfg.decision_hour].copy()
    df_dec["ticker"] = ticker

    drop_feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "logret_oc",
        "body",
        "upper_wick",
        "lower_wick",
        "mkt_Open",
        "mkt_High",
        "mkt_Low",
        "mkt_Close",
        "mkt_Volume",
        "mkt_logret_oc",
        "mkt_body",
        "mkt_upper_wick",
        "mkt_lower_wick",
    ]
    df_dec = df_dec.drop(columns=[c for c in drop_feature_cols if c in df_dec.columns])

    label_cols = ["y_ret", "y_risk"] if cfg.make_labels else []
    drop_cols = ["hour", "date"] + label_cols

    feature_cols = [
        c for c in df_dec.columns
        if c not in drop_cols and c != "ticker" and pd.api.types.is_numeric_dtype(df_dec[c])
    ]

    required = feature_cols + label_cols
    df_dec = df_dec.dropna(subset=required)

    return df_dec, feature_cols


def add_cross_sectional_ranks(df: pd.DataFrame, dt_col: str, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return df

    for c in use_cols:
        df[f"cs_rank_{c}"] = df.groupby(dt_col)[c].rank(pct=True, method="average")

    return df


# ---------------------------
# IO helpers
# ---------------------------

def load_preprocessed_parquet(prep_root: Path, ticker: str) -> Optional[pd.DataFrame]:
    p = prep_root / ticker / "intraday_1h_preprocessed.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    return df


def save_features_parquet(out_root: Path, ticker: str, df_feat: pd.DataFrame) -> Path:
    out_dir = out_root / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_1h_for_model1.parquet"
    df_feat.to_parquet(out_path, index=True)
    return out_path


# ---------------------------
# main (NO CLI)
# ---------------------------

def main() -> None:
    s = Settings()

    prep_root = s.prep_root.resolve()
    out_root = s.out_root.resolve()

    # TOPIX銘柄取得（.T付き）
    tickers: List[str] = list(extract_topix_codes())

    # 市場コードは確実に確保（特徴生成に必要）
    market_ticker = s.market_ticker
    df_mkt = load_preprocessed_parquet(prep_root, market_ticker)
    if df_mkt is None:
        raise FileNotFoundError(
            f"Market parquet not found: {prep_root / market_ticker / 'intraday_1h_preprocessed.parquet'}"
        )

    # 市場自身は通常「銘柄候補」に入れない
    tickers = [t for t in tickers if t != market_ticker]

    if s.limit_tickers and s.limit_tickers > 0:
        tickers = tickers[: s.limit_tickers]

    cfg = FeatureConfig(
        tz=s.tz,
        windows=s.windows,
        decision_hour=s.decision_hour,
        add_calendar=True,
        add_market=True,
        add_relative=True,
        make_labels=s.make_labels,
        label_type=s.label_type,
    )

    universe_tables: List[pd.DataFrame] = []
    use_sector = s.use_sector_features
    sector_map = _load_sector_map(s.sector_map_path) if use_sector else {}
    if use_sector and not sector_map:
        print(f"[WARN] sector_map not found or empty: {s.sector_map_path}")
        use_sector = False
    ok = 0
    missing = 0
    failed = 0

    for i, tkr in enumerate(tickers, start=1):
        df = load_preprocessed_parquet(prep_root, tkr)
        if df is None:
            missing += 1
            print(f"[WARN] ({i}/{len(tickers)}) missing parquet: {tkr}")
            continue

        try:
            feat_df, _ = build_features_for_ticker(
                df_ticker=df,
                df_market=df_mkt,
                cfg=cfg,
                ticker=tkr,
            )
            if use_sector or s.save_universe:
                universe_tables.append(feat_df)
            else:
                out_path = save_features_parquet(out_root, tkr, feat_df)
                ok += 1
                print(f"[OK] ({i}/{len(tickers)}) {tkr} -> {out_path} (rows={len(feat_df)})")

        except Exception as e:
            failed += 1
            print(f"[ERROR] ({i}/{len(tickers)}) failed: {tkr} :: {e}")

    print(f"\nDone. ok={ok}, missing={missing}, failed={failed}, requested={len(tickers)}")

    if universe_tables:
        uni = pd.concat(universe_tables, axis=0, ignore_index=False).sort_index()
        if use_sector:
            sector_proxy = build_sector_proxy_data(uni, sector_map, s.windows)
            uni = add_sector_features(uni, sector_proxy, s.windows)
        uni_reset = uni.reset_index()
        dt_name = uni_reset.columns[0]
        dup_count = uni_reset.duplicated(subset=[dt_name, "ticker"]).sum()
        if dup_count > 0:
            raise ValueError(f"Duplicate rows detected after feature build: {dt_name}+ticker dup={dup_count}")

        rank_cols = [
            "logret_cc",
            "ret_mean_24",
            "ret_mean_56",
            "ret_std_24",
            "ret_std_56",
            "dd_24",
            "dd_56",
            "rel_ret",
            "rel_range",
            "rel_vol",
            "v_rel_8",
            "v_rel_24",
            "v_rel_56",
            "volume_abs_mean_8",
            "volume_abs_mean_24",
            "volume_abs_mean_56",
            "volume_z_8",
            "volume_z_24",
            "volume_z_56",
            "volume_mom_8",
            "volume_mom_24",
            "volume_mom_56",
            "rsi_8",
            "rsi_24",
            "rsi_56",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "bb_z_8",
            "bb_z_24",
            "bb_z_56",
            "bb_width_8",
            "bb_width_24",
            "bb_width_56",
            "atr_pct_8",
            "atr_pct_24",
            "atr_pct_56",
            "adx_8",
            "adx_24",
            "adx_56",
            "obv_z_8",
            "obv_z_24",
            "obv_z_56",
            "sector_ret",
            "sector_rel_ret",
            "sector_vol",
            "sector_beta_24",
            "sector_beta_56",
            "sector_momentum_24",
            "sector_momentum_56",
        ]
        uni_reset = add_cross_sectional_ranks(uni_reset, dt_name, rank_cols)
        uni = uni_reset.set_index(dt_name)
        uni.index.name = dt_name

        if use_sector or s.save_universe:
            uni_dir = out_root / "universe"
            uni_dir.mkdir(parents=True, exist_ok=True)
            uni_path = uni_dir / "features_1h_for_model1_universe.parquet"
            uni.to_parquet(uni_path, index=True)
            print(f"[OK] universe saved: {uni_path} (rows={len(uni)})")

        if use_sector:
            for tkr, df_tkr in uni.groupby("ticker"):
                out_path = save_features_parquet(out_root, tkr, df_tkr)
                ok += 1
                print(f"[OK] {tkr} -> {out_path} (rows={len(df_tkr)})")


if __name__ == "__main__":
    main()
