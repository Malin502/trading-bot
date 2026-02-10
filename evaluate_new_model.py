#!/usr/bin/env python3
"""
新モデル（7 fold, single-task, 100特徴量）の性能評価
"""
import json
import numpy as np
from pathlib import Path

def load_results():
    """結果ファイルを読み込み"""
    with open("artifacts/model1/all_folds_metrics.json") as f:
        metrics = json.load(f)
    with open("artifacts/model1/all_folds_backtest.json") as f:
        backtest = json.load(f)
    return metrics, backtest

def extract_metrics(metrics, backtest):
    """メトリクスを抽出"""
    n_folds = len(metrics)
    
    # Validation結果
    val_ic = []
    val_sharpe = []
    val_mdd = []
    
    # Test結果
    test_ic = []
    test_sharpe = []
    test_mdd = []
    test_pf = []
    
    for i, m in enumerate(metrics):
        # Validation
        val_ic.append(m['val_backtest']['ic_ret'])
        val_sharpe.append(m['val_backtest']['sharpe_daily'])
        val_mdd.append(m['val_backtest']['max_drawdown'])
        
        # Test（backtest.jsonから）
        test = backtest['folds'][i]
        test_ic.append(test['ic_ret'])
        test_sharpe.append(test['sharpe_daily'])
        test_mdd.append(test['max_drawdown'])
        test_pf.append(test['pf'])
    
    return {
        'n_folds': n_folds,
        'val': {
            'ic': np.array(val_ic),
            'sharpe': np.array(val_sharpe),
            'mdd': np.array(val_mdd)
        },
        'test': {
            'ic': np.array(test_ic),
            'sharpe': np.array(test_sharpe),
            'mdd': np.array(test_mdd),
            'pf': np.array(test_pf)
        }
    }

def print_summary(results):
    """結果サマリーを出力"""
    print("=" * 80)
    print("📊 新モデル性能評価（7 fold, single-task, 100特徴量）")
    print("=" * 80)
    print()
    
    print(f"Fold数: {results['n_folds']}")
    print()
    
    # Validation
    print("### Validation期間（90日）")
    print("-" * 60)
    val = results['val']
    print(f"IC (return):    {val['ic'].mean():7.4f} ± {val['ic'].std():7.4f}  (範囲: {val['ic'].min():7.4f} ～ {val['ic'].max():7.4f})")
    print(f"Sharpe (daily): {val['sharpe'].mean():7.4f} ± {val['sharpe'].std():7.4f}  (範囲: {val['sharpe'].min():7.4f} ～ {val['sharpe'].max():7.4f})")
    print(f"Max Drawdown:   {val['mdd'].mean():7.2%} ± {val['mdd'].std():7.2%}  (最悪: {val['mdd'].min():7.2%})")
    print()
    
    # Test
    print("### Test期間（60日）")
    print("-" * 60)
    test = results['test']
    print(f"IC (return):    {test['ic'].mean():7.4f} ± {test['ic'].std():7.4f}  (範囲: {test['ic'].min():7.4f} ～ {test['ic'].max():7.4f})")
    print(f"Sharpe (daily): {test['sharpe'].mean():7.4f} ± {test['sharpe'].std():7.4f}  (範囲: {test['sharpe'].min():7.4f} ～ {test['sharpe'].max():7.4f})")
    print(f"Max Drawdown:   {test['mdd'].mean():7.2%} ± {test['mdd'].std():7.2%}  (最悪: {test['mdd'].min():7.2%})")
    print(f"Profit Factor:  {test['pf'].mean():7.4f} ± {test['pf'].std():7.4f}  (範囲: {test['pf'].min():7.4f} ～ {test['pf'].max():7.4f})")
    print()
    
    # Val-Test Gap
    print("### 汎化性能（Val-Test Gap）")
    print("-" * 60)
    ic_gap = val['ic'].mean() - test['ic'].mean()
    sharpe_gap = val['sharpe'].mean() - test['sharpe'].mean()
    print(f"IC Gap:     {ic_gap:+7.4f}  ({'悪化' if ic_gap > 0 else '改善'})")
    print(f"Sharpe Gap: {sharpe_gap:+7.4f}  ({'悪化' if sharpe_gap > 0 else '改善'})")
    print()
    
    # Fold別詳細
    print("### Fold別詳細（Test期間）")
    print("-" * 60)
    print("Fold | IC (ret) | Sharpe  | MDD     | PF    | 評価")
    print("-" * 60)
    for i in range(results['n_folds']):
        ic = test['ic'][i]
        sharpe = test['sharpe'][i]
        mdd = test['mdd'][i]
        pf = test['pf'][i]
        
        # 評価
        if sharpe > 0.05 and ic > 0.02:
            status = "✅ 良好"
        elif sharpe > 0:
            status = "⚠️ 要改善"
        else:
            status = "❌ 不良"
        
        print(f"  {i}  | {ic:8.4f} | {sharpe:7.4f} | {mdd:7.2%} | {pf:5.3f} | {status}")
    print()
    
    # 統計評価
    print("### 統計評価")
    print("-" * 60)
    
    # ICの有意性（t検定）
    from scipy import stats
    ic_tstat, ic_pval = stats.ttest_1samp(test['ic'], 0)
    print(f"IC有意性検定:")
    print(f"  t統計量: {ic_tstat:.4f}")
    print(f"  p値: {ic_pval:.4f}")
    if ic_pval < 0.05:
        print(f"  → ICは統計的に有意（5%水準）✅")
    else:
        print(f"  → ICは統計的に有意ではない ❌")
    print()
    
    # Sharpeのt検定
    sharpe_tstat, sharpe_pval = stats.ttest_1samp(test['sharpe'], 0)
    print(f"Sharpe有意性検定:")
    print(f"  t統計量: {sharpe_tstat:.4f}")
    print(f"  p値: {sharpe_pval:.4f}")
    if sharpe_pval < 0.05:
        if test['sharpe'].mean() > 0:
            print(f"  → Sharpeは統計的に正で有意（5%水準）✅")
        else:
            print(f"  → Sharpeは統計的に負で有意（5%水準）❌")
    else:
        print(f"  → Sharpeは統計的に有意ではない（ゼロと同等）⚠️")
    print()
    
    # 安定性
    ic_cv = test['ic'].std() / abs(test['ic'].mean()) if test['ic'].mean() != 0 else np.inf
    sharpe_cv = test['sharpe'].std() / abs(test['sharpe'].mean()) if test['sharpe'].mean() != 0 else np.inf
    print(f"変動係数（CV = std/mean）:")
    print(f"  IC CV:     {ic_cv:.2f}  ({'不安定' if ic_cv > 2 else '安定' if ic_cv < 1 else '中程度'})")
    print(f"  Sharpe CV: {sharpe_cv:.2f}  ({'不安定' if abs(sharpe_cv) > 2 else '安定' if abs(sharpe_cv) < 1 else '中程度'})")
    print()

def compare_with_old():
    """旧モデルとの比較"""
    print("=" * 80)
    print("📈 旧モデルとの比較")
    print("=" * 80)
    print()
    
    print("| 指標 | 旧モデル (5 fold) | 新モデル (7 fold) | 変化 | 評価 |")
    print("|------|------------------|------------------|------|------|")
    
    # 旧モデルの数値（ドキュメントから）
    old_test_ic = 0.033
    old_test_ic_std = 0.045
    old_test_sharpe = -0.043
    old_test_sharpe_std = 0.123
    old_test_mdd = -0.091
    
    # 新モデル
    metrics, backtest = load_results()
    results = extract_metrics(metrics, backtest)
    new_test_ic = results['test']['ic'].mean()
    new_test_ic_std = results['test']['ic'].std()
    new_test_sharpe = results['test']['sharpe'].mean()
    new_test_sharpe_std = results['test']['sharpe'].std()
    new_test_mdd = results['test']['mdd'].mean()
    
    # IC
    ic_change = new_test_ic - old_test_ic
    ic_pct = ic_change / abs(old_test_ic) * 100 if old_test_ic != 0 else 0
    ic_eval = "✅" if ic_change > 0 else "❌"
    print(f"| Test IC | {old_test_ic:.4f} ± {old_test_ic_std:.4f} | {new_test_ic:.4f} ± {new_test_ic_std:.4f} | {ic_change:+.4f} ({ic_pct:+.1f}%) | {ic_eval} |")
    
    # Sharpe
    sharpe_change = new_test_sharpe - old_test_sharpe
    sharpe_eval = "✅" if sharpe_change > 0 else "❌"
    print(f"| Test Sharpe | {old_test_sharpe:.4f} ± {old_test_sharpe_std:.4f} | {new_test_sharpe:.4f} ± {new_test_sharpe_std:.4f} | {sharpe_change:+.4f} | {sharpe_eval} |")
    
    # MDD
    mdd_change = new_test_mdd - old_test_mdd
    mdd_eval = "✅" if mdd_change > 0 else "❌"  # 数値が大きくなる=悪化
    print(f"| Test MDD | {old_test_mdd:.2%} | {new_test_mdd:.2%} | {mdd_change:+.2%} | {mdd_eval} |")
    
    print()
    print("**変更内容:**")
    print("- タスク: マルチタスク → シングルタスク（リターンのみ）")
    print("- 特徴量: 315個 → 100個（上位100選択）")
    print("- Fold数: 5 → 7（統計的信頼性向上）")
    print("- Walk-forward: train/val/test = 240/120/60 → 180/90/60")
    print()

def print_recommendations():
    """推奨事項"""
    print("=" * 80)
    print("💡 次のステップ")
    print("=" * 80)
    print()
    
    metrics, backtest = load_results()
    results = extract_metrics(metrics, backtest)
    
    test_ic_mean = results['test']['ic'].mean()
    test_sharpe_mean = results['test']['sharpe'].mean()
    test_ic_pval = 0.05  # 仮
    
    if test_sharpe_mean < 0:
        print("❌ **現状: 実運用不可**")
        print("   Test Sharpeがマイナス → 期待リターンが負")
        print()
    elif test_sharpe_mean < 0.1:
        print("⚠️ **現状: 実運用リスク大**")
        print("   Test Sharpeが低く、リスクに見合わない")
        print()
    else:
        print("✅ **現状: 実運用可能レベル**")
        print()
    
    print("### 🔥 最優先で実施すべき改善")
    print()
    
    if test_ic_mean < 0.03:
        print("1. **特徴量エンジニアリング**")
        print("   - IC < 0.03 → 予測力が不足")
        print("   - 重要度の低い特徴を更に削減（100 → 50）")
        print("   - 新しい特徴を追加（ファンダメンタル、センチメント等）")
        print()
    
    if test_sharpe_mean < 0:
        print("2. **ラベル定義の見直し**")
        print("   - 1日リターン → 3-5日リターンでノイズ削減")
        print("   - ラベル計算方法の検証（リーク確認）")
        print()
    
    print("3. **ベースラインモデルとの比較**")
    print("   - Ridge Regression / LightGBM で同じデータで評価")
    print("   - ニューラルネットの必要性を検証")
    print()
    
    print("4. **アンサンブル戦略の改善**")
    print("   - 性能の良いfoldのみ使用（IC > 0.02のfold）")
    print("   - ICベースの重み付け平均")
    print()
    
    print("### 🔶 中期的な改善")
    print()
    print("5. **データ拡充**")
    print("   - 学習期間を1200-1500日に延長（現在709日）")
    print("   - より多くの市場環境を学習")
    print()
    
    print("6. **ハイパーパラメータ最適化**")
    print("   - モデルサイズ（width, depth）")
    print("   - 学習率、Dropout率")
    print("   - 特徴量数")
    print()
    
    print("7. **セクター中立化**")
    print("   - セクター内でのロング・ショートでリスク削減")
    print()
    
    print("=" * 80)

def main():
    """メイン処理"""
    try:
        metrics, backtest = load_results()
        results = extract_metrics(metrics, backtest)
        
        print_summary(results)
        compare_with_old()
        print_recommendations()
        
        print("✅ 評価完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
