"""
topk最適化: k=3, 5, 10でPFとMDDを比較
最良のラベルタイプ (cc) を使用
"""
import subprocess
import json
import sys
import os
from pathlib import Path

TOPK_VALUES = [3, 5, 10]
BEST_LABEL = "cc"  # 引→翌日引

def run_topk_experiment(topk: int) -> dict:
    """指定topkで学習・評価を実行"""
    print(f"\n{'='*80}")
    print(f"実験開始: topk={topk}")
    print(f"{'='*80}\n")
    
    # 環境変数を設定
    env = os.environ.copy()
    env["TOPK"] = str(topk)
    
    # 1. モデル学習
    print(f"[1/2] モデル学習 (topk={topk})...")
    result = subprocess.run(
        ["python", "src/model1/train_model1.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERROR: 学習失敗\n{result.stderr}")
        return None
    print("完了\n")
    
    # 2. 全foldでバックテスト評価（環境変数でtopkを渡す）
    print(f"[2/2] 評価 (topk={topk})...")
    result = subprocess.run(
        ["python", "src/model1/eval_model1_folds.py"],
        capture_output=True,
        text=True,
        env=env
    )
    if result.returncode != 0:
        print(f"ERROR: 評価失敗\n{result.stderr}")
        return None
    print("完了\n")
    
    # 結果を読み込み
    backtest_path = Path("artifacts/model1/all_folds_backtest.json")
    if not backtest_path.exists():
        print(f"ERROR: {backtest_path} が見つかりません")
        return None
    
    with open(backtest_path, "r") as f:
        data = json.load(f)
    
    # aggキーの下に統計サマリがある
    if "agg" in data:
        result = data["agg"]
    else:
        result = data
    
    return result

def main():
    # まず、最良のラベル (cc) で特徴量を生成
    print(f"\n{'='*80}")
    print(f"最良ラベル ({BEST_LABEL}) で特徴量生成")
    print(f"{'='*80}\n")
    
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
    print("特徴量生成完了\n")
    
    # 各topk値で実験
    results = {}
    
    for topk in TOPK_VALUES:
        result = run_topk_experiment(topk)
        if result is not None:
            results[topk] = result
        else:
            print(f"警告: topk={topk} の実験がスキップされました")
    
    # 結果の比較表を出力
    print("\n" + "="*80)
    print("topk最適化結果サマリ")
    print("="*80)
    print(f"{'TopK':>6} {'Mean PF':>10} {'Mean LogRet':>12} {'Mean MDD':>10} {'Mean Sharpe':>12}")
    print("-"*80)
    
    for topk in TOPK_VALUES:
        if topk not in results:
            continue
        r = results[topk]
        mean_pf = r.get("mean_pf", float("nan"))
        mean_logret = r.get("mean_sum_logret", float("nan"))
        mean_mdd = r.get("mean_mdd", float("nan"))
        mean_sharpe = r.get("mean_sharpe", 0.0)
        
        print(f"{topk:>6} {mean_pf:>10.3f} {mean_logret:>12.4f} {mean_mdd:>10.4f} {mean_sharpe:>12.3f}")
    
    # 結果を保存
    output_path = Path("artifacts/model1/topk_comparison.json")
    with open(output_path, "w") as f:
        json.dump({
            "label_type": BEST_LABEL,
            "results": results
        }, f, indent=2)
    
    print(f"\n結果を保存: {output_path}")
    
    # 最良のtopkを推奨
    best_topk = None
    best_pf = -999
    for topk in TOPK_VALUES:
        if topk in results:
            pf = results[topk].get("mean_pf", -999)
            if pf > best_pf:
                best_pf = pf
                best_topk = topk
    
    if best_topk:
        print(f"\n推奨topk: {best_topk} - Mean PF={best_pf:.3f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
