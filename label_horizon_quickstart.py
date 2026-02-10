#!/usr/bin/env python3
"""
ラベル期間延長の実装ガイド - クイックスタート
"""

print("=" * 80)
print("📋 ラベル期間延長 - 実装チェックリスト")
print("=" * 80)
print()

print("### 🎯 目的")
print("- ノイズ削減によるIC向上（0.0135 → 0.03+）")
print("- Sharpe安定化（CV: 45 → 5以下）")
print("- 取引コスト削減（回転率: 0.75 → 0.3以下）")
print()

print("=" * 80)
print("### 📝 必要な修正箇所")
print("=" * 80)
print()

modifications = [
    {
        "file": "src/preprocessing/FeatureBuilder1h.py",
        "location": "class Settings (line ~24)",
        "change": "label_horizon パラメータを追加",
        "code": """
    # 既存フィールドの後に追加
    label_horizon: int = int(os.environ.get("LABEL_HORIZON", "1"))
""",
        "priority": "🔥 必須"
    },
    {
        "file": "src/preprocessing/FeatureBuilder1h.py",
        "location": "class FeatureConfig (line ~74)",
        "change": "label_horizon フィールドを追加",
        "code": """
    label_type: str = "cc"
    label_horizon: int = 1  # NEW
    risk_label_type: str = "drawdown"
""",
        "priority": "🔥 必須"
    },
    {
        "file": "src/preprocessing/FeatureBuilder1h.py",
        "location": "_make_daily_labels_from_hourly 関数 (line ~389)",
        "change": "label_horizon 引数を追加し、shift(-label_horizon)に変更",
        "code": """
def _make_daily_labels_from_hourly(
    df_ohlcv: pd.DataFrame,
    label_type: str = "oc",
    label_horizon: int = 1,  # NEW
    risk_label_type: str = "drawdown",
) -> pd.DataFrame:
    # ... 省略 ...
    
    # 変更箇所: shift(-1) → shift(-label_horizon)
    daily["next_open"] = daily["open_"].shift(-label_horizon)
    daily["next_close"] = daily["close_"].shift(-label_horizon)
    daily["next_low"] = daily["low_"].shift(-label_horizon)
    daily["next_high"] = daily["high_"].shift(-label_horizon)
    
    # 以降は従来通り
""",
        "priority": "🔥 必須"
    },
    {
        "file": "src/preprocessing/FeatureBuilder1h.py",
        "location": "build_features_for_ticker 関数内 (line ~460付近)",
        "change": "_make_daily_labels_from_hourly呼び出しにlabel_horizonを渡す",
        "code": """
    if cfg.make_labels:
        labels_df = _make_daily_labels_from_hourly(
            df_prep,
            label_type=cfg.label_type,
            label_horizon=cfg.label_horizon,  # NEW
            risk_label_type=cfg.risk_label_type,
        )
""",
        "priority": "🔥 必須"
    },
    {
        "file": "src/preprocessing/FeatureBuilder1h.py",
        "location": "run_all 関数内 (line ~650付近)",
        "change": "FeatureConfig生成時にlabel_horizonを渡す",
        "code": """
    cfg = FeatureConfig(
        tz=settings.tz,
        windows=settings.windows,
        decision_hour=settings.decision_hour,
        label_type=settings.label_type,
        label_horizon=settings.label_horizon,  # NEW
        risk_label_type=settings.risk_label_type,
        make_labels=settings.make_labels,
    )
""",
        "priority": "🔥 必須"
    },
]

for i, mod in enumerate(modifications, 1):
    print(f"{i}. {mod['priority']} {mod['file']}")
    print(f"   場所: {mod['location']}")
    print(f"   変更: {mod['change']}")
    print(f"   コード例:")
    for line in mod['code'].strip().split('\n'):
        print(f"   {line}")
    print()

print("=" * 80)
print("### 🧪 実験手順")
print("=" * 80)
print()

experiments = [
    {
        "name": "実験0: ベースライン確認",
        "horizon": 1,
        "command": "export LABEL_HORIZON=1",
        "purpose": "現状再現性確認",
        "expected_ic": "0.0135",
        "duration": "30分"
    },
    {
        "name": "実験1: 3日先予測",
        "horizon": 3,
        "command": "export LABEL_HORIZON=3",
        "purpose": "短期延長効果検証",
        "expected_ic": "0.020-0.025",
        "duration": "1時間"
    },
    {
        "name": "実験2: 5日先予測（1週間）",
        "horizon": 5,
        "command": "export LABEL_HORIZON=5",
        "purpose": "中期延長効果検証",
        "expected_ic": "0.025-0.035",
        "duration": "1時間"
    },
    {
        "name": "実験3: 10日先予測（2週間）",
        "horizon": 10,
        "command": "export LABEL_HORIZON=10",
        "purpose": "上限確認（長すぎる可能性）",
        "expected_ic": "0.015-0.025（期待薄）",
        "duration": "1時間"
    },
]

for i, exp in enumerate(experiments):
    print(f"{i+1}. {exp['name']}")
    print(f"   Horizon: {exp['horizon']}日")
    print(f"   コマンド:")
    print(f"      {exp['command']}")
    print(f"      python src/preprocessing/FeatureBuilder1h.py")
    print(f"      python src/model1/train_model1.py --max_folds 1")
    print(f"   目的: {exp['purpose']}")
    print(f"   期待IC: {exp['expected_ic']}")
    print(f"   所要時間: {exp['duration']}")
    print()

print("=" * 80)
print("### 📊 評価指標")
print("=" * 80)
print()

metrics = [
    ("IC (Information Coefficient)", "0.03以上", "予測力の指標"),
    ("Sharpe Ratio", "0.05以上", "リスク調整後リターン"),
    ("IC p値", "< 0.05", "統計的有意性"),
    ("Sharpe変動係数", "< 5.0", "安定性"),
    ("良好fold数", ">= 5/7", "汎化性能"),
]

print("| 指標 | 目標値 | 意味 |")
print("|------|--------|------|")
for metric, target, meaning in metrics:
    print(f"| {metric} | {target} | {meaning} |")

print()

print("=" * 80)
print("### ⚠️ 重要な注意点")
print("=" * 80)
print()

warnings = [
    ("Look-ahead Bias", 
     "特徴量生成時にy_retを参照しないこと",
     "実装後、特徴量日時 < ラベル日時 を確認"),
    
    ("データ欠損", 
     "horizon日分のデータが末尾で欠損",
     "horizon=5なら709日→704日に減少"),
    
    ("Walk-forward調整", 
     "test_daysをhorizon+αに設定",
     "test_days=60 → 65（horizon=5の場合）"),
    
    ("バックテスト調整",
     "リバランス頻度をhorizon日ごとに変更",
     "現在は毎日だが、horizon=3なら3日ごとに"),
]

for i, (title, issue, solution) in enumerate(warnings, 1):
    print(f"{i}. **{title}**")
    print(f"   問題: {issue}")
    print(f"   対策: {solution}")
    print()

print("=" * 80)
print("### 🎯 判断基準")
print("=" * 80)
print()

print("**実験1完了後（horizon=3）:**")
print()
print("✅ IC > 0.020  → Phase 2へ（全fold学習）")
print("⚠️ IC 0.015-0.020 → 実験2も実施（horizon=5）")
print("❌ IC < 0.015  → ラベル延長効果なし、別のアプローチ検討")
print()

print("**Phase 2完了後（全fold）:**")
print()
print("✅ 全条件達成 → Paper trading開始")
print("  - Test IC >= 0.03")
print("  - Test Sharpe >= 0.05")
print("  - IC有意性 p < 0.05")
print("  - 良好fold >= 5/7")
print()
print("⚠️ 一部未達 → ハイパーパラメータ調整")
print("❌ IC改善なし → 根本的な見直し")
print()

print("=" * 80)
print("### 🚀 クイックスタート")
print("=" * 80)
print()

print("1. 実装（今日）:")
print("   上記5箇所を修正")
print()

print("2. 実験1実行（今晩）:")
print("   export LABEL_HORIZON=3")
print("   python src/preprocessing/FeatureBuilder1h.py")
print("   python src/model1/train_model1.py --max_folds 1")
print()

print("3. 結果確認（明朝）:")
print("   python evaluate_new_model.py")
print("   → ICが0.02以上なら成功！")
print()

print("4. 次のステップ:")
print("   - horizon=5も試す")
print("   - 最良horizonで全fold学習")
print("   - Paper trading開始")
print()

print("=" * 80)
print("詳細は docs/label_horizon_strategy.md を参照")
print("=" * 80)
