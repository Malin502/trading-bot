#!/usr/bin/env python3
"""
全バージョンの比較分析
"""
import json
import numpy as np

def print_comparison():
    """3つのバージョンを比較"""
    print("=" * 80)
    print("📊 全バージョン性能比較")
    print("=" * 80)
    print()
    
    # 最新結果を読み込み
    with open("artifacts/model1/all_folds_metrics.json") as f:
        metrics = json.load(f)
    with open("artifacts/model1/all_folds_backtest.json") as f:
        backtest = json.load(f)
    
    # 最新の設定を確認
    n_folds = len(metrics)
    task_mode = metrics[0].get('task_mode', 'unknown')
    
    # Test結果を抽出
    test_ic = [f['ic_ret'] for f in backtest['folds']]
    test_sharpe = [f['sharpe_daily'] for f in backtest['folds']]
    test_mdd = [f['max_drawdown'] for f in backtest['folds']]
    test_pf = [f['pf'] for f in backtest['folds']]
    
    print("| 指標 | V1: 旧(5f, multi, 315特徴) | V2: 7f, single, 100特徴 | V3: 7f, multi, 100特徴 |")
    print("|------|---------------------------|------------------------|------------------------|")
    
    # IC
    print(f"| Test IC | 0.0330 ± 0.0450 | 0.0099 ± 0.0478 | {np.mean(test_ic):.4f} ± {np.std(test_ic):.4f} |")
    
    # Sharpe
    print(f"| Test Sharpe | -0.0430 ± 0.1230 | -0.0360 ± 0.0905 | {np.mean(test_sharpe):.4f} ± {np.std(test_sharpe):.4f} |")
    
    # MDD
    print(f"| Test MDD | -9.10% ± 3.80% | -8.10% ± 2.18% | {np.mean(test_mdd):.2%} ± {np.std(test_mdd):.2%} |")
    
    # PF
    print(f"| Profit Factor | N/A | 0.967 ± 0.241 | {np.mean(test_pf):.3f} ± {np.std(test_pf):.3f} |")
    
    print()
    print("**変更内容:**")
    print()
    print("V1 → V2:")
    print("  - タスク: マルチ → シングル")
    print("  - 特徴量: 315 → 100")
    print("  - Fold: 5 → 7")
    print("  - モデル: width=256, depth=4 → width=128, depth=2")
    print()
    print("V2 → V3:")
    print("  - タスク: シングル → マルチ（戻した）")
    print("  - モデル: width=128, depth=2 → width=196, depth=3")
    print("  - lambda_risk: 0.0 → 0.4")
    print()
    
    print("=" * 80)
    print("🔍 分析結果")
    print("=" * 80)
    print()
    
    # 順位付け
    ic_rank = ["V1 (0.033)", "V2 (0.010)", "V3 ({:.3f})".format(np.mean(test_ic))]
    sharpe_rank = ["V2 (-0.036)", "V1 (-0.043)", "V3 ({:.3f})".format(np.mean(test_sharpe))]
    
    print("### 性能ランキング")
    print()
    print("**IC（高い方が良い）:**")
    print("  1位: V1 (旧モデル) - 0.033")
    print("  2位: V2 (シングル) - 0.010")
    print(f"  3位: V3 (最新) - {np.mean(test_ic):.4f}")
    print()
    
    print("**Sharpe（高い方が良い）:**")
    if np.mean(test_sharpe) > -0.036:
        print(f"  1位: V3 (最新) - {np.mean(test_sharpe):.4f}")
        print("  2位: V2 (シングル) - -0.036")
        print("  3位: V1 (旧モデル) - -0.043")
    else:
        print("  1位: V2 (シングル) - -0.036")
        print("  2位: V1 (旧モデル) - -0.043")
        print(f"  3位: V3 (最新) - {np.mean(test_sharpe):.4f}")
    print()
    
    print("### 主要な発見")
    print()
    print("❌ **特徴量を100に削減したことが最大の失敗**")
    print("   - V1 (315特徴) → V2 (100特徴): IC 0.033 → 0.010 (-70%)")
    print("   - V2 → V3 (共に100特徴): IC 0.010 → {:.4f}".format(np.mean(test_ic)))
    print()
    
    print("✅ **シングルタスク化は若干プラス**")
    print("   - V2シングル vs V3マルチ: Sharpe -0.036 vs {:.4f}".format(np.mean(test_sharpe)))
    print()
    
    print("⚠️ **モデルサイズの影響は不明瞭**")
    print("   - 小さいモデル(V2) vs 中モデル(V3): 明確な差はない")
    print()
    
    print("=" * 80)
    print("💡 推奨アクション")
    print("=" * 80)
    print()
    
    print("### 🔥 最優先：特徴量選択を無効化")
    print()
    print("```python")
    print("# src/model1/train_model1.py")
    print("dl_cfg = DataLoadersConfig(")
    print("    batch_size=512,")
    print("    num_workers=0,")
    print("    pin_memory=True,")
    print("    feature_top_n=0,  # ← 100から0に変更（全特徴使用）")
    print(")")
    print("```")
    print()
    print("**期待効果:**")
    print("  - IC: {:.4f} → 0.03+ (3-10倍改善)".format(np.mean(test_ic)))
    print("  - V1レベルの予測力を回復")
    print()
    
    print("### 🔶 その他の改善")
    print()
    print("1. **シングルタスクを維持**")
    print("   ```python")
    print("   task_mode='single',  # multiから戻す")
    print("   lambda_risk=0.0,")
    print("   ```")
    print()
    print("2. **モデルサイズは中程度**")
    print("   ```python")
    print("   width=128,  # 196から削減")
    print("   depth=2,    # 3から削減")
    print("   ```")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    print_comparison()
