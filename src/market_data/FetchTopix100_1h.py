from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

TZ = "Asia/Tokyo"


# ---------- 正規化 / クリーニング ----------

def normalize_intraday_1h(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance history() の結果を、
    - indexをJSTへ統一
    - 1時間床丸め
    - 重複除去（last）
    - 必要列のみ(Open/High/Low/Close/Volume)
    に正規化する
    """
    if df is None or df.empty:
        return pd.DataFrame()

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # tz統一
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ)
    else:
        df.index = df.index.tz_convert(TZ)

    # 1時間に床丸め（時刻ズレ吸収）
    df.index = df.index.floor("1h")

    # 重複除去（同一timestampは最後を採用）
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 型
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def slice_to_1500_jst(df: pd.DataFrame) -> pd.DataFrame:
    """
    全期間について「各営業日 09:00〜15:00（15:00含む）」だけ残す。
    15:00以降が混ざる問題を恒久的に排除する。
    """
    if df.empty:
        return df

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    else:
        idx = idx.tz_convert(TZ)

    # 時刻条件：09:00 <= t <= 15:00
    t = idx.time
    mask = (idx.hour > 9) | ((idx.hour == 9) & (idx.minute >= 0))
    # ↑は冗長なので、素直に時刻でフィルタする
    mask = (idx.strftime("%H:%M:%S") >= "09:00:00") & (idx.strftime("%H:%M:%S") <= "15:00:00")
    out = df.copy()
    out.index = idx
    return out.loc[mask].sort_index()


# ---------- 保存（月Parquet upsert） ----------

def upsert_parquet_monthly(out_dir: Path, symbol: str, df: pd.DataFrame) -> None:
    """
    clean/{symbol}/intraday_1h_YYYY-MM.parquet に月単位で追記（重複はlastで潰す）
    """
    if df.empty:
        return

    sym_dir = out_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    # 月ごとに分割して保存
    # to_period("M") は tz付きでもOK (警告は無視して問題なし)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for m, df_m in df.groupby(df.index.to_period("M")):
            yyyy_mm = str(m)  # "2026-02"
            path = sym_dir / f"intraday_1h_{yyyy_mm}.parquet"

            if path.exists():
                old = pd.read_parquet(path)

                # 旧データのtzが落ちてたら補正
                if isinstance(old.index, pd.DatetimeIndex):
                    if old.index.tz is None:
                        old.index = old.index.tz_localize(TZ)
                    else:
                        old.index = old.index.tz_convert(TZ)

                merged = pd.concat([old, df_m], axis=0)
            else:
                merged = df_m

            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_parquet(path)


# ---------- 取得（上限730d） ----------

@dataclass
class FetchConfig:
    interval: str = "1h"
    chunk_period: str = "730d"   # yfinanceの上限に合わせる
    sleep_sec: float = 1.0
    max_retries: int = 3
    max_workers: int = 10  # 並列スレッド数（APIレート制限を考慮）


def fetch(symbol: str, cfg: FetchConfig, end_dt: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    end_dt を基準に period=730d, interval=1h のチャンクを取る
    """
    last_err: Optional[Exception] = None

    # yfinanceに渡す end は tzなしのほうが無難なので日付文字列にする
    end_str = None
    if end_dt is not None:
        if end_dt.tzinfo is not None:
            end_dt = end_dt.tz_convert(TZ)
        end_str = end_dt.strftime("%Y-%m-%d")

    for i in range(cfg.max_retries):
        try:
            t = yf.Ticker(symbol)
            df = t.history(
                period=cfg.chunk_period,
                interval=cfg.interval,
                auto_adjust=False
            )
            return df
        except Exception as e:
            last_err = e
            time.sleep(cfg.sleep_sec * (i + 1))

    raise RuntimeError(f"fetch failed: {symbol} end={end_str}") from last_err


def _fetch_single_symbol(sym: str, cfg: FetchConfig, now_jst: pd.Timestamp, out_clean: Path) -> tuple[str, bool, str]:
    """
    単一銘柄のデータ取得と保存（スレッド内で実行）
    Returns: (symbol, success, message)
    """
    try:
        all_parts = []

        raw = fetch(sym, cfg, end_dt=now_jst)

        part = normalize_intraday_1h(raw)
        part = slice_to_1500_jst(part)  # 15:15などを全期間で排除

        if not part.empty:
            all_parts.append(part)

        time.sleep(cfg.sleep_sec)

        if all_parts:
            df = pd.concat(all_parts, axis=0)
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = pd.DataFrame()

        upsert_parquet_monthly(out_clean, sym, df)

        last_ts = df.index.max() if not df.empty else None
        msg = f"rows={len(df)} last={last_ts}"
        return (sym, True, msg)

    except Exception as e:
        return (sym, False, str(e))


def fetch_and_save_730d(symbols: Iterable[str], out_clean_dir: str = "data/clean") -> None:
    """
    symbols（.T付き）を対象に、730dまでのデータを取得して保存する（並列処理版）
    """
    cfg = FetchConfig()
    out_clean = Path(out_clean_dir)
    out_clean.mkdir(parents=True, exist_ok=True)

    # 基準となる end（今日）
    now_jst = pd.Timestamp.now(tz=TZ)

    symbols = [s.strip() for s in symbols if s and s.strip()]

    print(f"取得開始: {len(symbols)}銘柄を{cfg.max_workers}スレッドで並列処理")
    
    success_count = 0
    failure_count = 0

    # ThreadPoolExecutorで並列処理
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        # 全銘柄のタスクを投入
        future_to_symbol = {
            executor.submit(_fetch_single_symbol, sym, cfg, now_jst, out_clean): sym 
            for sym in symbols
        }

        # 完了したタスクから順次結果を取得
        for future in as_completed(future_to_symbol):
            sym, success, msg = future.result()
            if success:
                print(f"[OK] {sym}: {msg}")
                success_count += 1
            else:
                print(f"[NG] {sym}: {msg}")
                failure_count += 1

    print(f"\n完了: 成功={success_count} 失敗={failure_count}")


# -------- 使い方 --------
# topix100 = ["7203.T", "6758.T", ...]  # もう.T付きのリスト
# fetch_and_save_240d(topix100, out_clean_dir="data/clean")

if __name__ == "__main__":
    # テスト用コード抽出・取得実行
    import sys
    from pathlib import Path
    
    # プロジェクトルートをsys.pathに追加
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.market_data.CodeExtractor import extract_topix_codes

    topix100 = extract_topix_codes()
    
    # 市場コード（1306.T）も追加
    market_ticker = "1306.T"
    if market_ticker not in topix100:
        topix100 = [market_ticker] + list(topix100)
    
    print(f"取得したコード数: {len(topix100)} (市場コード含む)")

    fetch_and_save_730d(topix100, out_clean_dir="data/clean")