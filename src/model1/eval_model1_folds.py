# scripts/eval_model1_folds.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from backtest import BacktestConfig, run_topk_daily_backtest


# ============================
# Settings (WRITE HERE)
# ============================
PROJECT_ROOT = Path(__file__).parent.parent.parent

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts/model1"
TOPK = int(os.environ.get("TOPK", "5"))
EPS = 1e-6
SCORE_CLIP = None      # 例: 50.0 / Noneなら無効
SAVE_DAILY = True


# ============================
# Helpers
# ============================

def find_fold_dirs(artifacts_root: Path) -> List[Path]:
    return sorted([p for p in artifacts_root.glob("fold_*") if p.is_dir()])


def load_preds_test(fold_dir: Path) -> pd.DataFrame:
    p = fold_dir / "preds_test.parquet"
    if not p.exists():
        raise FileNotFoundError(f"preds_test.parquet not found: {p}")

    df = pd.read_parquet(p)

    # datetime列名の吸収
    if "datetime" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        else:
            raise ValueError(f"No datetime column found in {p}. columns={list(df.columns)}")

    if "ticker" not in df.columns:
        # tickerが別名ならここで吸収しても良い
        raise ValueError(f"No ticker column found in {p}. columns={list(df.columns)}")

    return df


# ============================
# Main
# ============================

def main():
    artifacts_root = ARTIFACTS_ROOT
    fold_dirs = find_fold_dirs(artifacts_root)
    if not fold_dirs:
        raise SystemExit(f"No fold dirs under: {artifacts_root}")

    cfg = BacktestConfig(
        topk=TOPK,
        eps=EPS,
        score_clip=SCORE_CLIP,
    )

    all_rows: List[Dict[str, Any]] = []

    print(f"Backtest start: root={artifacts_root}, topk={TOPK}, eps={EPS}, score_clip={SCORE_CLIP}")

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name  # fold_021
        try:
            fold_id = int(fold_name.split("_")[1])
        except Exception:
            fold_id = -1

        df = load_preds_test(fold_dir)
        out = run_topk_daily_backtest(df, cfg=cfg, datetime_col="datetime", ticker_col="ticker")

        summary = out["summary"]
        summary_row = {"fold": fold_id, **summary}
        all_rows.append(summary_row)

        # foldごと保存
        (fold_dir / "backtest_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if SAVE_DAILY:
            out["daily"].to_parquet(fold_dir / "backtest_daily.parquet", index=False)

        print(
            f"[{fold_name}] "
            f"sum_logret={summary['sum_logret']:.6f} "
            f"pf={summary['pf']:.3f} "
            f"mdd={summary['max_drawdown']:.3f} "
            f"rankic={summary['rankic_mean']:.3f} "
            f"ic_ret={summary['ic_ret']:.3f}"
        )

    df_all = pd.DataFrame(all_rows).sort_values("fold").reset_index(drop=True)

    # 全fold集計
    # PFがinfになるケースは平均計算から除外
    pf_series = df_all["pf"].replace([float("inf")], pd.NA).dropna()

    agg = {
        "n_folds": int(len(df_all)),
        "topk": int(TOPK),
        "mean_sum_logret": float(df_all["sum_logret"].mean()),
        "median_sum_logret": float(df_all["sum_logret"].median()),
        "mean_pf": float(pf_series.mean()) if len(pf_series) else 0.0,
        "mean_mdd": float(df_all["max_drawdown"].mean()),
        "mean_rankic": float(df_all["rankic_mean"].mean()),
        "mean_ic_ret": float(df_all["ic_ret"].mean()),
    }

    # まとめて保存（csv/json）
    (artifacts_root / "all_folds_backtest.csv").write_text(df_all.to_csv(index=False), encoding="utf-8")
    (artifacts_root / "all_folds_backtest.json").write_text(
        json.dumps({"agg": agg, "folds": all_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Aggregate ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
