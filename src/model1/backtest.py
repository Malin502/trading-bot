# src/model1/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------
# Settings (EDIT HERE)
# ---------------------------

# 入力予測ファイル（preds_test.parquet など）
INPUT_PATH = Path("artifacts/model1/fold_006/preds_test.parquet")

# 日次PnLの保存先（不要なら None）
SAVE_DAILY_PATH: Optional[Path] = None

@dataclass
class BacktestConfig:
    topk: int = 5  # 日次で選抜する銘柄数
    eps: float = 1e-6  # 分母ゼロ回避のための微小値
    score_clip: Optional[float] = None  # スコアの上下クリップ（Noneなら無効）
    score_method: str = "sharpe_adj"  # スコア方式: ret_only | simple | cost_aware | sharpe_adj | utility
    risk_aversion: float = 1.0  # utilityのリスク回避係数（無次元、大きいほど保守的）
    cost_bps: float = 0.0  # 取引コスト（bps、往復。10bps=0.10%）
    slippage_bps: float = 5.0  # スリッページ（bps、往復。10bps=0.10%）
    turnover_penalty_bps: float = 8.0  # 前日ポジ差分ベースの選抜時ペナルティ（bps, 片道）
    sector_neutralize: bool = False  # セクター中立化（セクター内でスコアをランク正規化）
    sector_map_path: Optional[Path] = None  # セクターマッピングJSONのパス
    

# バックテスト設定
CFG = BacktestConfig(
    topk=5,  # 日次で選ぶ銘柄数
    score_method="utility",  # スコア方式（利益重視ならutility推奨）
    risk_aversion=1.0,  # リスク回避（無次元、utilityのみ有効）
    cost_bps=0.0,  # 取引コスト（bps、往復。10bps=0.10%）
    slippage_bps=5.0,  # スリッページ（bps、往復。10bps=0.10%）
    score_clip=None,  # スコアの上下クリップ
)

def _to_date(s: pd.Series) -> pd.Series:
    # datetime列 or DatetimeIndex のどちらでも扱えるようにする
    dt = pd.to_datetime(s)
    return dt.dt.date


def add_score(
    df: pd.DataFrame,
    eps: float = 1e-6,
    score_clip: Optional[float] = None,
    method: str = "simple",
    risk_aversion: float = 1.0,
    sector_neutralize: bool = False,
    sector_map: Optional[Dict[str, str]] = None,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    out = df.copy()
    out["pred_risk_pos"] = out["pred_risk"].clip(lower=0.0)

    if method == "ret_only":
        score = out["pred_ret"]
    elif method == "utility":
        score = out["pred_ret"] - risk_aversion * (out["pred_risk_pos"] ** 2)
    elif method == "sharpe_adj":
        score = out["pred_ret"] / (out["pred_risk_pos"] + eps) - 0.5 * out["pred_risk_pos"]
    elif method == "cost_aware":
        score = out["pred_ret"] / (out["pred_risk_pos"] + eps)
    else:
        score = out["pred_ret"] / (out["pred_risk_pos"] + eps)

    out["score"] = score
    if score_clip is not None and score_clip > 0:
        out["score"] = out["score"].clip(-score_clip, score_clip)
    
    # セクター中立化: セクター内でスコアをランク正規化
    if sector_neutralize and sector_map is not None:
        out["sector"] = out[ticker_col].map(sector_map)
        # セクターが割り当てられている銘柄のみ処理
        mask = out["sector"].notna()
        if mask.sum() > 0:
            out.loc[mask, "score_rank"] = out.loc[mask].groupby("sector")["score"].rank(pct=True)
            # ランクを[-1, 1]にスケール（0.5を中央として）
            out.loc[mask, "score"] = 2 * (out.loc[mask, "score_rank"] - 0.5)
            out.drop(columns=["score_rank"], inplace=True)
        out.drop(columns=["sector"], inplace=True, errors="ignore")
    
    return out


def equity_from_logrets(logrets: np.ndarray) -> np.ndarray:
    # log return 累積 → equity
    return np.exp(np.cumsum(logrets))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def profit_factor_from_logrets(logrets: np.ndarray) -> float:
    """
    PFは本来「利益/損失」なので単純リターンで計算する。
    daily simple return = exp(logret) - 1
    PF = sum(pos) / abs(sum(neg))
    """
    if len(logrets) == 0:
        return 0.0
    r = np.exp(logrets) - 1.0
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def sharpe_from_logrets(logrets: np.ndarray) -> float:
    if len(logrets) < 2:
        return 0.0
    mu = float(np.mean(logrets))
    sd = float(np.std(logrets, ddof=1))
    if sd == 0:
        return 0.0
    # 日次シャープ（年率化しない。必要なら *sqrt(252)）
    return float(mu / sd)


def daily_rankic_spearman(df_day: pd.DataFrame, col_x: str, col_y: str) -> float:
    """
    Spearman = rank相関（pandasのrankで近似）
    """
    if len(df_day) < 3:
        return np.nan
    x = df_day[col_x].rank(method="average")
    y = df_day[col_y].rank(method="average")
    vx = x.to_numpy(dtype=np.float64)
    vy = y.to_numpy(dtype=np.float64)
    sx = vx.std()
    sy = vy.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(vx, vy)[0, 1])


def run_topk_daily_backtest(
    df_pred: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
    datetime_col: str = "datetime",
    ticker_col: str = "ticker",
) -> Dict[str, Any]:
    """
    df_pred 必須列：
      - datetime_col
      - ticker_col
      - y_ret (実現 log return)
      - pred_ret
      - pred_risk
    """
    required = [datetime_col, ticker_col, "y_ret", "pred_ret", "pred_risk"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise ValueError(f"df_pred missing columns: {missing}")

    df = df_pred.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values([datetime_col, ticker_col]).reset_index(drop=True)
    
    # セクターマッピングを読み込み
    sector_map = None
    if cfg.sector_neutralize and cfg.sector_map_path is not None:
        import json
        with open(cfg.sector_map_path, "r") as f:
            sector_map = json.load(f)

    df = add_score(
        df,
        eps=cfg.eps,
        score_clip=cfg.score_clip,
        method=cfg.score_method,
        risk_aversion=cfg.risk_aversion,
        sector_neutralize=cfg.sector_neutralize,
        sector_map=sector_map,
        ticker_col=ticker_col,
    )
    df["date"] = _to_date(df[datetime_col])

    # 日次TopK選抜と日次PnL（gross/net log return）
    daily_rows = []
    rankic_list = []
    prev_w: Dict[str, float] = {}
    one_way_cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0 / 2.0

    for d, g in df.groupby("date", sort=True):
        g_work = g.copy()
        if cfg.turnover_penalty_bps > 0 and cfg.topk > 0:
            target_w = 1.0 / float(cfg.topk)
            prev_w_ser = g_work[ticker_col].astype(str).map(prev_w).fillna(0.0)
            delta_buy = (target_w - prev_w_ser).clip(lower=0.0)
            penalty = delta_buy * (cfg.turnover_penalty_bps / 10000.0)
            g_work["score"] = g_work["score"] - penalty

        g2 = g_work.sort_values("score", ascending=False).head(cfg.topk)

        # 同一日内の重複tickerがあっても壊れないようにticker単位へ集約
        if len(g2):
            g2_u = g2.groupby(ticker_col, as_index=False).agg(y_ret=("y_ret", "mean"))
            gross_logret = float(g2_u["y_ret"].mean())
            n = int(len(g2_u))
            curr_w = {t: 1.0 / n for t in g2_u[ticker_col].astype(str).tolist()}
        else:
            gross_logret = 0.0
            n = 0
            curr_w = {}

        keys = set(prev_w.keys()) | set(curr_w.keys())
        buy_turnover = float(sum(max(0.0, curr_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))
        sell_turnover = float(sum(max(0.0, prev_w.get(k, 0.0) - curr_w.get(k, 0.0)) for k in keys))
        turnover = 0.5 * (buy_turnover + sell_turnover)

        # 売買回転率ベースで片道コストを控除（初日の新規建てにも対応）
        trading_cost_rate = (buy_turnover + sell_turnover) * one_way_cost_rate
        gross_simple = float(np.exp(gross_logret) - 1.0)
        net_simple = gross_simple - trading_cost_rate
        net_simple = max(net_simple, -0.999999)  # log1pの定義域保護
        net_logret = float(np.log1p(net_simple))

        daily_rows.append(
            {
                "date": d,
                "n": n,
                "daily_logret_gross": gross_logret,
                "daily_turnover": turnover,
                "daily_trading_cost_rate": trading_cost_rate,
                "daily_logret": net_logret,
            }
        )
        prev_w = curr_w

        # RankIC（score と y_ret の相関を日次で）
        rankic = daily_rankic_spearman(g_work, "score", "y_ret")
        rankic_list.append(rankic)

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    logrets_net = daily["daily_logret"].to_numpy(dtype=np.float64)
    logrets_gross = daily["daily_logret_gross"].to_numpy(dtype=np.float64)

    eq_net = equity_from_logrets(logrets_net)
    eq_gross = equity_from_logrets(logrets_gross)
    mdd_net = max_drawdown(eq_net)
    mdd_gross = max_drawdown(eq_gross)
    pf_net = profit_factor_from_logrets(logrets_net)
    pf_gross = profit_factor_from_logrets(logrets_gross)
    sharpe_net = sharpe_from_logrets(logrets_net)
    sharpe_gross = sharpe_from_logrets(logrets_gross)

    # 追加の整合指標（IC：全体相関）
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3:
            return 0.0
        sa = a.std()
        sb = b.std()
        if sa == 0 or sb == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    ic_ret = _corr(df["pred_ret"].to_numpy(np.float64), df["y_ret"].to_numpy(np.float64))
    # riskは y_risk があれば見る（なければスキップ）
    ic_risk = None
    if "y_risk" in df.columns:
        ic_risk = _corr(df["pred_risk"].to_numpy(np.float64), df["y_risk"].to_numpy(np.float64))

    rankic_mean = float(np.nanmean(np.array(rankic_list, dtype=np.float64))) if len(rankic_list) else 0.0

    summary = {
        "topk": cfg.topk,
        "n_days": int(len(daily)),
        # 互換キー: 既存利用側はこの値を参照し続けられる（net）
        "sum_logret": float(logrets_net.sum()) if len(logrets_net) else 0.0,
        "mean_daily_logret": float(logrets_net.mean()) if len(logrets_net) else 0.0,
        "pf": pf_net,
        "max_drawdown": mdd_net,
        "sharpe_daily": sharpe_net,
        # 明示キー
        "sum_logret_net": float(logrets_net.sum()) if len(logrets_net) else 0.0,
        "sum_logret_gross": float(logrets_gross.sum()) if len(logrets_gross) else 0.0,
        "mean_daily_logret_net": float(logrets_net.mean()) if len(logrets_net) else 0.0,
        "mean_daily_logret_gross": float(logrets_gross.mean()) if len(logrets_gross) else 0.0,
        "pf_net": pf_net,
        "pf_gross": pf_gross,
        "max_drawdown_net": mdd_net,
        "max_drawdown_gross": mdd_gross,
        "sharpe_daily_net": sharpe_net,
        "sharpe_daily_gross": sharpe_gross,
        "avg_daily_turnover": float(daily["daily_turnover"].mean()) if len(daily) else 0.0,
        "avg_daily_trading_cost_rate": float(daily["daily_trading_cost_rate"].mean()) if len(daily) else 0.0,
        "ic_ret": ic_ret,
        "ic_risk": ic_risk,
        "rankic_mean": rankic_mean,
    }

    return {
        "summary": summary,
        "daily": daily,
    }


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print("Backtest Summary")
    print("=" * 80)
    print(f"topk: {summary['topk']}")
    print(f"n_days: {summary['n_days']}")
    print(f"sum_logret_net: {summary['sum_logret_net']:.6f}")
    print(f"sum_logret_gross: {summary['sum_logret_gross']:.6f}")
    print(f"mean_daily_logret_net: {summary['mean_daily_logret_net']:.6f}")
    print(f"mean_daily_logret_gross: {summary['mean_daily_logret_gross']:.6f}")
    print(f"pf_net: {summary['pf_net']:.4f}")
    print(f"pf_gross: {summary['pf_gross']:.4f}")
    print(f"max_drawdown_net: {summary['max_drawdown_net']:.4f}")
    print(f"max_drawdown_gross: {summary['max_drawdown_gross']:.4f}")
    print(f"sharpe_daily_net: {summary['sharpe_daily_net']:.4f}")
    print(f"sharpe_daily_gross: {summary['sharpe_daily_gross']:.4f}")
    print(f"avg_daily_turnover: {summary['avg_daily_turnover']:.4f}")
    print(f"avg_daily_trading_cost_rate: {summary['avg_daily_trading_cost_rate']:.6f}")
    print(f"ic_ret: {summary['ic_ret']:.4f}")
    ic_risk = summary.get("ic_risk", None)
    if ic_risk is not None:
        print(f"ic_risk: {ic_risk:.4f}")
    print(f"rankic_mean: {summary['rankic_mean']:.4f}")


def main() -> None:
    df = _load_predictions(INPUT_PATH)
    out = run_topk_daily_backtest(df, cfg=CFG)
    _print_summary(out["summary"])

    if SAVE_DAILY_PATH is not None:
        SAVE_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
        out["daily"].to_csv(SAVE_DAILY_PATH, index=False)
        print(f"Saved daily PnL: {SAVE_DAILY_PATH}")


if __name__ == "__main__":
    main()
