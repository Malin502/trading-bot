# trade/update_features_model1.py
"""
モデル① 特徴量 増分更新（max rolling=56h）

毎営業日（引け後）に、モデル①用の特徴量テーブルを更新する。
rolling 特徴量（最大窓幅56時間）を正しく計算するため、必要な過去データだけを参照しつつ、
出力は「直近営業日 1日分（15:00決定の1行/銘柄）」のみ追記/置換する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 既存の特徴量計算ロジックを再利用
from src.preprocessing.FeatureBuilder1h import (
    FeatureConfig,
    add_sector_features,
    add_cross_sectional_normalization,
    build_features_for_ticker,
    build_sector_proxy_data,
    load_preprocessed_parquet,
)
from src.market_data.CodeExtractor import extract_topix_codes


# ============================
# Settings (EDIT HERE)
# ============================

# 前処理済み1時間足データのルート
PREPROCESS_ROOT = PROJECT_ROOT / "data/preprocessing"

# 出力先（推論用：当日のみ）
INFERENCE_OUT_PARQUET = PROJECT_ROOT / "features/universe/features_1h_for_model1_latest_1d.parquet"

# 市場コード（特徴量計算に必要）
MARKET_TICKER = "1306.T"

# セクターマップ
SECTOR_MAP_PATH = PROJECT_ROOT / "config/sector_mapping.json"

# 必要な過去履歴の範囲（1時間足ベース）
# 最大rolling窓幅56h + 安全マージン24h = 80本
LOOKBACK_BARS = 80

# タイムゾーン
TZ = "Asia/Tokyo"

# 決定時刻（15:00の行を抽出）
DECISION_HOUR = 15

# rolling窓幅設定（学習時と同じ）
WINDOWS = (8, 24, 56)

# ログレベル（True: 詳細ログ、False: エラーのみ）
VERBOSE = True


# ============================
# Helpers
# ============================

def get_universe_tickers() -> List[str]:
    """Universe の銘柄一覧を取得（TOPIX銘柄）"""
    tickers = list(extract_topix_codes())
    # 市場コード自身は除外
    tickers = [t for t in tickers if t != MARKET_TICKER]
    return tickers


def load_intraday_1h(ticker: str, prep_root: Path) -> Optional[pd.DataFrame]:
    """1時間足の前処理済みデータをロード"""
    return load_preprocessed_parquet(prep_root, ticker)


def slice_lookback(df_1h: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    """
    直近 lookback_bars 本だけを取得
    rolling計算用の参照データ
    """
    if len(df_1h) <= lookback_bars:
        return df_1h.copy()
    return df_1h.iloc[-lookback_bars:].copy()


def compute_features_for_ticker_lookback(
    df_ticker: pd.DataFrame,
    df_market: pd.DataFrame,
    ticker: str,
    cfg: FeatureConfig,
) -> Optional[pd.DataFrame]:
    """
    1銘柄の特徴量を計算し、決定時刻（15:00）の行を返す
    
    Returns:
        pd.DataFrame: index=DatetimeIndex（1行）、columns=[ticker, 特徴量...]
        None: 計算失敗時
    """
    try:
        # 既存の build_features_for_ticker を使用（決定時刻フィルタ済み）
        feat_df, _ = build_features_for_ticker(
            df_ticker=df_ticker,
            df_market=df_market,
            cfg=cfg,
            ticker=ticker,
        )
        
        if len(feat_df) == 0:
            return None

        return feat_df
        
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] compute_features_for_one_day({ticker}): {e}")
        return None





def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """parquet 保存（ディレクトリも作成）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


# ============================
# Main
# ============================

def main() -> None:
    print("=" * 80)
    print("モデル① 推論用特徴量生成（当日のみ）")
    print("=" * 80)
    
    # 1. 設定準備
    cfg = FeatureConfig(
        tz=TZ,
        windows=WINDOWS,
        decision_hour=DECISION_HOUR,
        add_calendar=True,
        add_market=True,
        add_relative=True,
        make_labels=False,  # 推論用更新ではラベル不要
    )
    
    # 2. 市場データをロード（特徴量計算に必要）
    print(f"[INFO] Loading market data: {MARKET_TICKER}")
    df_mkt = load_intraday_1h(MARKET_TICKER, PREPROCESS_ROOT)
    if df_mkt is None:
        raise FileNotFoundError(
            f"Market parquet not found: {PREPROCESS_ROOT / MARKET_TICKER / 'intraday_1h_preprocessed.parquet'}"
        )
    
    # 市場データも直近分だけに絞る
    df_mkt = slice_lookback(df_mkt, LOOKBACK_BARS)
    print(f"[INFO] Market data loaded: {len(df_mkt)} bars")
    
    # 3. 銘柄一覧を取得
    tickers = get_universe_tickers()
    print(f"[INFO] Universe tickers: {len(tickers)}")
    
    # 4. 各銘柄の特徴量を計算（lookback分）
    new_rows_list: List[pd.DataFrame] = []
    ok_count = 0
    missing_count = 0
    failed_count = 0
    
    for i, ticker in enumerate(tickers, start=1):
        # 1時間足データをロード
        df_1h = load_intraday_1h(ticker, PREPROCESS_ROOT)
        if df_1h is None:
            missing_count += 1
            if VERBOSE:
                print(f"[WARN] ({i}/{len(tickers)}) Missing data: {ticker}")
            continue
        
        # 直近 LOOKBACK_BARS 本だけに絞る
        df_1h_recent = slice_lookback(df_1h, LOOKBACK_BARS)
        
        # 特徴量計算（決定時刻行の時系列）
        feat_df = compute_features_for_ticker_lookback(
            df_ticker=df_1h_recent,
            df_market=df_mkt,
            ticker=ticker,
            cfg=cfg,
        )
        
        if feat_df is None or len(feat_df) == 0:
            failed_count += 1
            if VERBOSE:
                print(f"[WARN] ({i}/{len(tickers)}) Failed to compute: {ticker}")
            continue

        new_rows_list.append(feat_df)
        ok_count += 1
        
        if VERBOSE and i % 10 == 0:
            print(f"[INFO] ({i}/{len(tickers)}) Processed: {ticker} -> {feat_df.index[-1].date()}")
    
    # 5. 結果の統合
    if not new_rows_list:
        print("[ERROR] No features computed. Exiting without saving.")
        return
    
    result_df = pd.concat(new_rows_list, axis=0, ignore_index=False).sort_index()

    sector_map = {}
    if SECTOR_MAP_PATH.exists():
        sector_map = json.loads(SECTOR_MAP_PATH.read_text(encoding="utf-8"))
    if sector_map:
        try:
            sector_proxy = build_sector_proxy_data(result_df, sector_map, WINDOWS)
            result_df = add_sector_features(result_df, sector_proxy, WINDOWS)
        except Exception as e:
            if VERBOSE:
                print(f"[WARN] sector features skipped: {e}")

    # Cross-sectional normalization (rank/zscore)
    result_reset = result_df.reset_index()
    dt_name = result_reset.columns[0]
    result_reset = add_cross_sectional_normalization(
        result_reset,
        dt_col=dt_name,
        exclude_cols=[dt_name, "ticker"],
        add_rank=True,
        add_zscore=True,
    )
    result_df = result_reset.set_index(dt_name)

    result_reset = result_df.reset_index()
    dt_name = result_reset.columns[0]
    dup_all = result_reset.duplicated(subset=[dt_name, "ticker"]).sum()
    if dup_all > 0:
        raise ValueError(f"Duplicate rows detected before latest-day filter: {dt_name}+ticker dup={dup_all}")

    latest_date = result_df.index.date.max()
    result_df = result_df.loc[result_df.index.date == latest_date].copy()

    latest_reset = result_df.reset_index()
    dup_latest = latest_reset.duplicated(subset=[latest_reset.columns[0], "ticker"]).sum()
    if dup_latest > 0:
        raise ValueError(f"Duplicate rows detected on latest day: dup={dup_latest}")
    
    print(f"\n[INFO] Computed rows: {len(result_df)}")
    print(f"[INFO] Date: {result_df.index.date.min()} (latest trading day)")
    print(f"[INFO] Tickers: {len(result_df['ticker'].unique())}")
    
    # 6. 保存（当日データのみ上書き）
    save_parquet(result_df, INFERENCE_OUT_PARQUET)
    print(f"[OK] Inference features saved: {INFERENCE_OUT_PARQUET}")
    
    # 7. サマリー
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  OK:      {ok_count}")
    print(f"  Missing: {missing_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Total:   {len(tickers)}")
    print("=" * 80)
    
    # 8. 失敗率が高すぎる場合は警告
    fail_ratio = (missing_count + failed_count) / len(tickers) if tickers else 0.0
    if fail_ratio > 0.5:
        print(f"[WARN] High failure ratio: {fail_ratio:.1%}")
        print("[WARN] Please check data availability.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
