"""
複数のラベルタイプ（寄→引, 引→引, 寄→寄）で学習・評価を比較
"""
import subprocess
import json
import sys
import os
from pathlib import Path

LABEL_TYPES = ["oc", "cc", "oo"]
LABEL_NAMES = {
    "oc": "翌日寄→引",
    "cc": "引→翌日引", 
    "oo": "寄→翌日寄"
}

def run_experiment(label_type: str) -> dict:
    """指定ラベルタイプで特徴量生成→学習→評価を実行"""
    print(f"\n{'='*80}")
    print(f"実験開始: label_type={label_type} ({LABEL_NAMES[label_type]})")
    print(f"{'='*80}\n")
    
    # 環境変数を設定
    env = os.environ.copy()
    env["LABEL_TYPE"] = label_type
    
    # 1. 特徴量生成
    print(f"[1/3] 特徴量生成 (label_type={label_type})...")
    result = subprocess.run(
        ["python", "src/preprocessing/FeatureBuilder1h.py"],
        capture_output=True,
        text=True,
        env=env
    )
    if result.returncode != 0:
        print(f"ERROR: 特徴量生成失敗\n{result.stderr}")
        return None
    print("完了\n")
    
    # 2. モデル学習
    print(f"[2/3] モデル学習...")
    result = subprocess.run(
        ["python", "src/model1/train_model1.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERROR: 学習失敗\n{result.stderr}")
        return None
    print("完了\n")
    
    # 3. 全foldでバックテスト評価
    print(f"[3/3] 評価...")
    result = subprocess.run(
        ["python", "src/model1/eval_model1_folds.py"],
        capture_output=True,
        text=True
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
    results = {}
    
    for label_type in LABEL_TYPES:
        result = run_experiment(label_type)
        if result is not None:
            results[label_type] = result
        else:
            print(f"警告: {label_type} の実験がスキップされました")
    
    # 結果の比較表を出力
    print("\n" + "="*80)
    print("ラベル比較結果サマリ")
    print("="*80)
    print(f"{'Label Type':<12} {'Name':<15} {'Mean PF':>10} {'Mean LogRet':>12} {'Mean MDD':>10} {'Mean Sharpe':>12}")
    print("-"*80)
    
    for label_type in LABEL_TYPES:
        if label_type not in results:
            continue
        r = results[label_type]
        name = LABEL_NAMES[label_type]
        mean_pf = r.get("mean_pf", float("nan"))
        mean_logret = r.get("mean_sum_logret", float("nan"))
        mean_mdd = r.get("mean_mdd", float("nan"))
        mean_sharpe = r.get("mean_sharpe", float("nan"))
        
        print(f"{label_type:<12} {name:<15} {mean_pf:>10.3f} {mean_logret:>12.4f} {mean_mdd:>10.4f} {mean_sharpe:>12.3f}")
    
    # 結果を保存
    output_path = Path("artifacts/model1/label_comparison.json")
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "label_names": LABEL_NAMES
        }, f, indent=2)
    
    print(f"\n結果を保存: {output_path}")
    
    # 最良のラベルタイプを推奨
    best_label = None
    best_pf = -999
    for label_type in LABEL_TYPES:
        if label_type in results:
            pf = results[label_type].get("mean_pf", -999)
            if pf > best_pf:
                best_pf = pf
                best_label = label_type
    
    if best_label:
        print(f"\n推奨ラベル: {best_label} ({LABEL_NAMES[best_label]}) - Mean PF={best_pf:.3f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
