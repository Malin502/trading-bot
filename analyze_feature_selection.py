#!/usr/bin/env python3
"""
特徴量選択の妥当性を検証
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

def check_selected_features():
    """各foldで選択された特徴を確認"""
    print("=" * 80)
    print("🔍 特徴量選択の分析")
    print("=" * 80)
    print()
    
    # 各foldのモデル情報を確認
    for fold_id in range(7):
        fold_dir = Path(f"artifacts/model1/fold_{fold_id:03d}")
        
        # 特徴量リストファイルがあるか確認
        features_file = fold_dir / "selected_features.json"
        if features_file.exists():
            with open(features_file) as f:
                features = json.load(f)
            print(f"Fold {fold_id}: {len(features)} 特徴選択済み")
            print(f"  上位10: {features[:10]}")
            print()
        else:
            print(f"Fold {fold_id}: 特徴量リストファイルなし")
            print()
    
    print("=" * 80)

def check_fold_data():
    """各foldのデータ期間を確認"""
    print("=" * 80)
    print("📅 各Foldのデータ期間")
    print("=" * 80)
    print()
    
    with open("artifacts/model1/all_folds_metrics.json") as f:
        metrics = json.load(f)
    
    print("Fold | Train日数 | Val日数 | Test日数")
    print("-" * 50)
    
    for fold in metrics:
        fold_id = fold['fold']
        val_days = fold['val_backtest']['n_days']
        test_days = fold['test_backtest']['n_days']
        
        print(f"  {fold_id}  |    ?      |   {val_days}    |   {test_days}")
    
    print()
    print("=" * 80)

def compare_feature_selection_methods():
    """特徴量選択方法の比較提案"""
    print("=" * 80)
    print("💡 特徴量選択の改善案")
    print("=" * 80)
    print()
    
    print("現在の方法:")
    print("  |corr(feature, y_ret)| の上位N個")
    print()
    
    print("問題点:")
    print("  1. 単変量の相関のみで、特徴量間の相互作用を無視")
    print("  2. 非線形な関係を捉えられない")
    print("  3. 過去データへの過適合リスク")
    print()
    
    print("改善案:")
    print("  A. より多くの特徴を保持（100 → 150-200）")
    print("  B. LightGBMの特徴量重要度を使用")
    print("  C. LASSO回帰で自動選択")
    print("  D. Permutation Importance")
    print("  E. 特徴量選択なし（全315特徴使用）")
    print()
    
    print("=" * 80)

def main():
    try:
        check_selected_features()
        check_fold_data()
        compare_feature_selection_methods()
        
        print()
        print("✅ 分析完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
