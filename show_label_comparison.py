"""ラベル比較結果を表示"""
import json
from pathlib import Path

LABEL_NAMES = {
    "oc": "翌日寄→引",
    "cc": "引→翌日引", 
    "oo": "寄→翌日寄"
}

comparison_path = Path("artifacts/model1/label_comparison.json")

with open(comparison_path, "r") as f:
    data = json.load(f)

results = data["results"]

print("\n" + "="*80)
print("ラベル比較結果サマリ")
print("="*80)
print(f"{'Label Type':<12} {'Name':<15} {'Mean PF':>10} {'Mean LogRet':>12} {'Mean MDD':>10} {'Mean Sharpe':>12}")
print("-"*80)

for label_type in ["oc", "cc", "oo"]:
    if label_type not in results:
        continue
    r = results[label_type]["agg"]
    name = LABEL_NAMES[label_type]
    mean_pf = r.get("mean_pf", float("nan"))
    mean_logret = r.get("mean_sum_logret", float("nan"))
    mean_mdd = r.get("mean_mdd", float("nan"))
    mean_sharpe = r.get("mean_sharpe", 0.0)
    
    print(f"{label_type:<12} {name:<15} {mean_pf:>10.3f} {mean_logret:>12.4f} {mean_mdd:>10.4f} {mean_sharpe:>12.3f}")

# 最良のラベルタイプを推奨
best_label = None
best_pf = -999
for label_type in ["oc", "cc", "oo"]:
    if label_type in results:
        pf = results[label_type]["agg"].get("mean_pf", -999)
        if pf > best_pf:
            best_pf = pf
            best_label = label_type

print(f"\n推奨ラベル: {best_label} ({LABEL_NAMES[best_label]}) - Mean PF={best_pf:.3f}")

# 各ラベルの詳細統計
print("\n" + "="*80)
print("詳細統計")
print("="*80)
for label_type in ["oc", "cc", "oo"]:
    if label_type not in results:
        continue
    r = results[label_type]["agg"]
    name = LABEL_NAMES[label_type]
    print(f"\n{label_type} ({name}):")
    print(f"  Mean PF:      {r.get('mean_pf', 0.0):.4f}")
    print(f"  Mean LogRet:  {r.get('mean_sum_logret', 0.0):.6f}")
    print(f"  Median LogRet:{r.get('median_sum_logret', 0.0):.6f}")
    print(f"  Mean MDD:     {r.get('mean_mdd', 0.0):.6f}")
    print(f"  Mean RankIC:  {r.get('mean_rankic', 0.0):.6f}")
    print(f"  Mean IC(ret): {r.get('mean_ic_ret', 0.0):.6f}")
