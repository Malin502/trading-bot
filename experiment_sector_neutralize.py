"""
セクター中立化の効果を検証
最良の設定（label=cc, topk=5）でセクター中立化あり/なしを比較
"""
import subprocess
import json
import sys
import os
from pathlib import Path

BEST_LABEL = "cc"  # 引→翌日引
BEST_TOPK = 5
PROJECT_ROOT = Path(__file__).parent
SECTOR_MAP_PATH = PROJECT_ROOT / "config/sector_mapping.json"

def run_sector_experiment(sector_neutralize: bool) -> dict:
    """セクター中立化あり/なしで評価を実行"""
    label = "有効" if sector_neutralize else "無効"
    print(f"\n{'='*80}")
    print(f"実験開始: セクター中立化={label}")
    print(f"{'='*80}\n")
    
    # eval_model1_folds.pyを修正してセクター中立化を有効にする必要がある
    # 環境変数で渡す
    env = os.environ.copy()
    env["TOPK"] = str(BEST_TOPK)
    env["SECTOR_NEUTRALIZE"] = "1" if sector_neutralize else "0"
    
    # 評価実行
    print(f"評価 (sector_neutralize={sector_neutralize})...")
    
    # 一時的な評価スクリプトを作成
    eval_script = f"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.model1.backtest import BacktestConfig, run_topk_daily_backtest

# 設定
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts/model1"
TOPK = {BEST_TOPK}
EPS = 1e-6
SCORE_CLIP = None
SECTOR_NEUTRALIZE = {str(sector_neutralize)}
SECTOR_MAP_PATH = PROJECT_ROOT / "config/sector_mapping.json"

def evaluate_fold(fold_idx: int, sector_neutralize: bool):
    fold_dir = ARTIFACTS_ROOT / f"fold_{{fold_idx:03d}}"
    test_preds = fold_dir / "preds_test.parquet"
    
    if not test_preds.exists():
        return None
    
    df = pd.read_parquet(test_preds)
    
    cfg = BacktestConfig(
        topk=TOPK,
        eps=EPS,
        score_clip=SCORE_CLIP,
        score_method="utility",
        risk_aversion=1.0,
        cost_bps=5.0,
        slippage_bps=3.0,
        sector_neutralize=sector_neutralize,
        sector_map_path=SECTOR_MAP_PATH if sector_neutralize else None,
    )
    
    out = run_topk_daily_backtest(df, cfg=cfg, datetime_col="datetime", ticker_col="ticker")
    return out["summary"]

# 全foldで評価
fold_results = []
for fold_idx in range(22):
    result = evaluate_fold(fold_idx, SECTOR_NEUTRALIZE)
    if result is not None:
        fold_results.append(result)

# 集計
if fold_results:
    summary = {{
        "n_folds": len(fold_results),
        "topk": TOPK,
        "mean_sum_logret": sum(r["sum_logret"] for r in fold_results) / len(fold_results),
        "median_sum_logret": sorted([r["sum_logret"] for r in fold_results])[len(fold_results) // 2],
        "mean_pf": sum(r["pf"] for r in fold_results) / len(fold_results),
        "mean_mdd": sum(r["max_drawdown"] for r in fold_results) / len(fold_results),
        "mean_rankic": sum(r.get("rankic_mean", 0.0) for r in fold_results) / len(fold_results),
        "mean_ic_ret": sum(r.get("ic_ret", 0.0) for r in fold_results) / len(fold_results),
        "mean_sharpe": sum(r.get("sharpe_daily", 0.0) for r in fold_results) / len(fold_results),
    }}
    
    # 結果を保存
    output = {{
        "agg": summary,
        "folds": fold_results
    }}
    
    output_path = ARTIFACTS_ROOT / "sector_neutralize_{{}}_{{}}.json".format(
        "enabled" if SECTOR_NEUTRALIZE else "disabled", TOPK
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(json.dumps(summary, indent=2))
"""
    
    script_path = f"temp_eval_sector_{int(sector_neutralize)}.py"
    with open(script_path, "w") as f:
        f.write(eval_script)
    
    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        env=env
    )
    
    # クリーンアップ
    Path(script_path).unlink(missing_ok=True)
    
    if result.returncode != 0:
        print(f"ERROR: 評価失敗\n{result.stderr}")
        return None
    print("完了\n")
    
    # 結果を読み込み
    status = "enabled" if sector_neutralize else "disabled"
    result_path = PROJECT_ROOT / f"artifacts/model1/sector_neutralize_{status}_{BEST_TOPK}.json"
    if not result_path.exists():
        print(f"ERROR: {result_path} が見つかりません")
        return None
    
    with open(result_path, "r") as f:
        data = json.load(f)
    
    return data["agg"]

def main():
    # 最良の設定で特徴量が既に生成されていることを前提とする
    # 必要なら再生成
    env_check = os.environ.get("LABEL_TYPE")
    if env_check != BEST_LABEL:
        print(f"\n特徴量を再生成 (label_type={BEST_LABEL})...")
        env = os.environ.copy()
        env["LABEL_TYPE"] = BEST_LABEL
        result = subprocess.run(
            ["python", "src/preprocessing/FeatureBuilder1h.py"],
            capture_output=True,
            text=True,
            env=env
        )
        if result.returncode != 0:
            print(f"ERROR: 特徴量生成失敗\n{result.stderr}")
            return 1
        print("完了\n")
    
    # モデルが既に学習済みであることを前提とする
    # 必要なら再学習
    if not (PROJECT_ROOT / "artifacts/model1/fold_000/preds_test.parquet").exists():
        print("\nモデルを再学習...")
        result = subprocess.run(
            ["python", "src/model1/train_model1.py"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"ERROR: 学習失敗\n{result.stderr}")
            return 1
        print("完了\n")
    
    results = {}
    
    # セクター中立化なし
    result = run_sector_experiment(sector_neutralize=False)
    if result is not None:
        results["disabled"] = result
    
    # セクター中立化あり
    result = run_sector_experiment(sector_neutralize=True)
    if result is not None:
        results["enabled"] = result
    
    # 結果の比較表を出力
    print("\n" + "="*80)
    print("セクター中立化比較結果")
    print("="*80)
    print(f"{'Status':<12} {'Mean PF':>10} {'Mean LogRet':>12} {'Mean MDD':>10} {'Mean Sharpe':>12}")
    print("-"*80)
    
    for status in ["disabled", "enabled"]:
        if status not in results:
            continue
        r = results[status]
        label = "無効" if status == "disabled" else "有効"
        mean_pf = r.get("mean_pf", float("nan"))
        mean_logret = r.get("mean_sum_logret", float("nan"))
        mean_mdd = r.get("mean_mdd", float("nan"))
        mean_sharpe = r.get("mean_sharpe", 0.0)
        
        print(f"{label:<12} {mean_pf:>10.3f} {mean_logret:>12.6f} {mean_mdd:>10.4f} {mean_sharpe:>12.3f}")
    
    # 結果を保存
    output_path = PROJECT_ROOT / "artifacts/model1/sector_neutralize_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "label_type": BEST_LABEL,
            "topk": BEST_TOPK,
            "results": results
        }, f, indent=2)
    
    print(f"\n結果を保存: {output_path}")
    
    # 推奨
    if "enabled" in results and "disabled" in results:
        pf_enabled = results["enabled"].get("mean_pf", 0.0)
        pf_disabled = results["disabled"].get("mean_pf", 0.0)
        if pf_enabled > pf_disabled:
            print(f"\n推奨: セクター中立化を有効にする (Mean PF: {pf_disabled:.3f} → {pf_enabled:.3f})")
        else:
            print(f"\n推奨: セクター中立化は無効のまま (Mean PF: {pf_disabled:.3f} > {pf_enabled:.3f})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
