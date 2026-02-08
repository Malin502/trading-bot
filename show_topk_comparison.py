"""
topk最適化結果を表示
"""
import json
from pathlib import Path

comparison_path = Path("artifacts/model1/topk_comparison.json")

with open(comparison_path, "r") as f:
    data = json.load(f)

results = data["results"]
label_type = data["label_type"]

print("\n" + "="*80)
print(f"topk最適化結果サマリ (label_type={label_type})")
print("="*80)
print(f"{'TopK':>6} {'Mean PF':>10} {'Mean LogRet':>12} {'Mean MDD':>10} {'Mean Sharpe':>12}")
print("-"*80)

for topk in [3, 5, 10]:
    if str(topk) not in results:
        continue
    r = results[str(topk)]
    mean_pf = r.get("mean_pf", float("nan"))
    mean_logret = r.get("mean_sum_logret", float("nan"))
    mean_mdd = r.get("mean_mdd", float("nan"))
    mean_sharpe = r.get("mean_sharpe", 0.0)
    
    print(f"{topk:>6} {mean_pf:>10.3f} {mean_logret:>12.6f} {mean_mdd:>10.4f} {mean_sharpe:>12.3f}")

# 最良のtopkを推奨
best_topk = None
best_pf = -999
for topk in [3, 5, 10]:
    if str(topk) in results:
        pf = results[str(topk)].get("mean_pf", -999)
        if pf > best_pf:
            best_pf = pf
            best_topk = topk

print(f"\n推奨topk: {best_topk} - Mean PF={best_pf:.3f}")

# 詳細統計
print("\n" + "="*80)
print("詳細統計")
print("="*80)
for topk in [3, 5, 10]:
    if str(topk) not in results:
        continue
    r = results[str(topk)]
    print(f"\ntopk={topk}:")
    print(f"  Mean PF:      {r.get('mean_pf', 0.0):.4f}")
    print(f"  Mean LogRet:  {r.get('mean_sum_logret', 0.0):.6f}")
    print(f"  Median LogRet:{r.get('median_sum_logret', 0.0):.6f}")
    print(f"  Mean MDD:     {r.get('mean_mdd', 0.0):.6f}")
    print(f"  Mean RankIC:  {r.get('mean_rankic', 0.0):.6f}")
    print(f"  Mean IC(ret): {r.get('mean_ic_ret', 0.0):.6f}")
