# ラベル期間延長の実装方針

**作成日**: 2026-02-10  
**目的**: ノイズ削減によるIC向上、Sharpe改善

---

## 📊 現状分析

### 現在のラベル定義
```python
# FeatureBuilder1h.py: _make_daily_labels_from_hourly

label_type = "cc"  # 引→翌日引（1日リターン）

# 実装
daily["next_close"] = daily["close_"].shift(-1)  # 1日先
daily["y_ret"] = np.log(next_close / close_)
```

### 問題点
1. **短期ノイズが大きい**
   - 1日リターンは市場ノイズの影響が大
   - IC = 0.0135（低い）
   - Sharpe変動係数 = 45.18（非常に不安定）

2. **予測困難性**
   - 日次レベルの価格変動は予測困難
   - マイクロストラクチャーノイズ
   - オーバーナイト・ギャップの影響

3. **高回転率**
   - 1日予測 → 毎日リバランス
   - 回転率 0.7-0.9 → 取引コスト大

---

## 🎯 ラベル期間延長の目的

### 主目的
1. **ノイズ削減による予測精度向上**
   - Signal-to-Noise比の改善
   - IC: 0.0135 → 0.03-0.05（目標）

2. **Sharpe Ratioの安定化**
   - 短期変動の平準化
   - 統計的有意性の獲得

3. **取引コスト削減**
   - 保有期間延長 → 回転率低下
   - 0.7-0.9 → 0.3-0.5（目標）

### 副次的効果
- 過学習リスク低減
- より安定したアルファ獲得
- 実運用の実現可能性向上

---

## 💡 実装方針

### A. パラメータ設計

#### 新パラメータ: `label_horizon`
```python
@dataclass
class Settings:
    # ... existing fields ...
    
    # ラベル期間（営業日ベース）
    label_horizon: int = os.environ.get("LABEL_HORIZON", 1)
    # 1: 1日先（現在）
    # 3: 3日先
    # 5: 5日先（1週間）
    # 10: 10日先（2週間）
```

#### 実装例
```python
def _make_daily_labels_from_hourly(
    df_ohlcv: pd.DataFrame,
    label_type: str = "cc",
    label_horizon: int = 1,  # NEW
    risk_label_type: str = "drawdown",
) -> pd.DataFrame:
    """
    label_horizon: 何営業日先のリターンを計算するか
      1: 翌日（従来）
      3: 3日先
      5: 5日先（1週間）
     10: 10日先（2週間）
    """
    df = df_ohlcv.copy()
    df["date"] = df.index.date

    daily = df.groupby("date").agg(
        open_=("Open", "first"),
        close_=("Close", "last"),
        low_=("Low", "min"),
        high_=("High", "max"),
    )

    # label_horizon日先の価格を取得
    daily["next_open"] = daily["open_"].shift(-label_horizon)
    daily["next_close"] = daily["close_"].shift(-label_horizon)
    daily["next_low"] = daily["low_"].shift(-label_horizon)
    daily["next_high"] = daily["high_"].shift(-label_horizon)

    # ラベル計算（従来と同じロジック）
    if label_type == "cc":
        # 引 → horizon日後の引
        daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["close_"] + EPS))
    elif label_type == "oc":
        # horizon日後の寄 → 引
        daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["next_open"] + EPS))
    elif label_type == "oo":
        # 寄 → horizon日後の寄
        daily["y_ret"] = np.log((daily["next_open"] + EPS) / (daily["open_"] + EPS))
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    # リスクラベル（同様に期間延長）
    if risk_label_type == "abs_ret":
        daily["y_risk"] = daily["y_ret"].abs()
    elif risk_label_type == "drawdown":
        # horizon日間の最大ドローダウン
        # 簡易版: next_lowを使用（正確にはrolling minが必要）
        if label_type == "cc":
            base = daily["close_"]
        elif label_type == "oc":
            base = daily["next_open"]
        else:
            base = daily["open_"]
        daily["y_risk"] = np.maximum(0.0, (base - daily["next_low"]) / (base + EPS))
    elif risk_label_type == "intraday_range":
        daily["y_risk"] = (daily["next_high"] - daily["next_low"]) / (daily["next_open"] + EPS)
        daily["y_risk"] = daily["y_risk"].clip(lower=0.0)
    else:
        raise ValueError(f"Unknown risk_label_type: {risk_label_type}")

    daily["y_risk"] = daily["y_risk"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    return daily[["y_ret", "y_risk"]]
```

---

## 🧪 実験計画（段階的アプローチ）

### Phase 1: 基礎検証（1週間）

#### 実験1: horizon=3（3日先）
```bash
# 環境変数で設定
export LABEL_HORIZON=3
export LABEL_TYPE=cc

# 特徴量再生成
python src/preprocessing/FeatureBuilder1h.py

# 1 fold のみ学習（クイック検証）
python src/model1/train_model1.py --max_folds 1
```

**評価指標**:
- IC改善度: 0.0135 → 0.02+ を期待
- Sharpe改善度
- Val-Test Gap縮小

**判断基準**:
- IC > 0.02: Phase 2へ進む ✅
- 0.015-0.02: 微妙、horizon=5も試す ⚠️
- < 0.015: horizon延長の効果なし ❌

---

#### 実験2: horizon=5（5日先・1週間）
```bash
export LABEL_HORIZON=5
export LABEL_TYPE=cc

python src/preprocessing/FeatureBuilder1h.py
python src/model1/train_model1.py --max_folds 1
```

**期待効果**:
- IC: 0.02-0.04
- Sharpe: 0.05-0.10
- より安定

**注意点**:
- サンプル数が減る（最後の5日が欠損）
- 長期すぎると予測が困難に

---

#### 実験3: horizon=10（10日先・2週間）
```bash
export LABEL_HORIZON=10

python src/preprocessing/FeatureBuilder1h.py
python src/model1/train_model1.py --max_folds 1
```

**目的**: 上限確認
- horizonを長くしすぎると逆効果の可能性
- 10日はおそらく長すぎる

---

### Phase 2: 最適horizon決定（3-5日）

最もICが高かったhorizonで**全7 fold学習**

```bash
# 仮にhorizon=3が最良だった場合
export LABEL_HORIZON=3
python src/preprocessing/FeatureBuilder1h.py
python src/model1/train_model1.py  # 全fold
```

**評価**:
- 全fold平均IC
- Sharpe（統計的有意性）
- 安定性（CV）
- fold間のばらつき

**成功基準**:
- Test IC >= 0.03 ✅
- Test Sharpe >= 0.05 ✅
- IC有意性 p < 0.05 ✅
- 良好fold >= 5/7 ✅

---

### Phase 3: 取引戦略調整（1週間）

#### バックテストパラメータの調整

現在の設定:
```python
topk = 5  # 上位5銘柄保有
リバランス: 毎日
```

horizon=3-5にした場合:
```python
topk = 5  # そのまま
リバランス: 3-5日ごと（horizon期間）
保有期間: 3-5日
```

**期待効果**:
- 取引コスト削減: 60-80%減
- 実質Sharpe向上

---

## 📋 推奨実装手順

### Step 1: コード修正（1日）

#### 1-1. Settings拡張
```python
# src/preprocessing/FeatureBuilder1h.py

@dataclass
class Settings:
    # 既存フィールド...
    
    # NEW: ラベル期間（営業日）
    label_horizon: int = int(os.environ.get("LABEL_HORIZON", "1"))
    
    # デフォルトは1（従来通り）
```

#### 1-2. _make_daily_labels_from_hourly修正
```python
def _make_daily_labels_from_hourly(
    df_ohlcv: pd.DataFrame,
    label_type: str = "cc",
    label_horizon: int = 1,  # NEW
    risk_label_type: str = "drawdown",
) -> pd.DataFrame:
    # 上記の実装例を参照
    pass
```

#### 1-3. build_features_for_ticker修正
```python
def build_features_for_ticker(
    ticker: str,
    cfg: FeatureConfig,
    # ... 他のパラメータ
) -> pd.DataFrame:
    # ...
    if cfg.make_labels:
        labels_df = _make_daily_labels_from_hourly(
            df_prep,
            label_type=cfg.label_type,
            label_horizon=cfg.label_horizon,  # NEW
            risk_label_type=cfg.risk_label_type,
        )
    # ...
```

#### 1-4. FeatureConfig更新
```python
@dataclass
class FeatureConfig:
    # 既存...
    label_type: str = "cc"
    label_horizon: int = 1  # NEW
    risk_label_type: str = "drawdown"
```

---

### Step 2: 実験実行（3-5日）

```bash
# 実験管理スクリプト
#!/bin/bash

for horizon in 1 3 5 10; do
    echo "=== Testing LABEL_HORIZON=$horizon ==="
    
    export LABEL_HORIZON=$horizon
    export LABEL_TYPE=cc
    
    # 特徴量生成
    python src/preprocessing/FeatureBuilder1h.py
    
    # 1 fold学習
    python src/model1/train_model1.py --max_folds 1
    
    # 結果保存
    mv artifacts/model1 artifacts/model1_horizon${horizon}
    
    echo "Done: horizon=$horizon"
done

# 結果比較
python compare_horizons.py
```

---

### Step 3: 結果分析と最適化（2-3日）

#### 比較スクリプト作成
```python
# compare_horizons.py

import json
import pandas as pd

results = []
for horizon in [1, 3, 5, 10]:
    path = f"artifacts/model1_horizon{horizon}/all_folds_metrics.json"
    with open(path) as f:
        metrics = json.load(f)
    
    # Fold 0のみ比較（クイック検証）
    val_ic = metrics[0]['val_backtest']['ic_ret']
    test_ic = metrics[0]['test_backtest']['ic_ret']
    test_sharpe = metrics[0]['test_backtest']['sharpe_daily']
    
    results.append({
        'horizon': horizon,
        'val_ic': val_ic,
        'test_ic': test_ic,
        'test_sharpe': test_sharpe,
    })

df = pd.DataFrame(results)
print(df)
print()
print("Best IC:", df.loc[df['test_ic'].idxmax()])
```

---

## ⚠️ 注意点とリスク

### 1. データ欠損の増加
```
現在: 最後の1日が欠損（709 → 708日）
horizon=3: 最後の3日が欠損（709 → 706日）
horizon=5: 最後の5日が欠損（709 → 704日）
```

**対策**: データ期間が短いため、horizon=5までに留める

---

### 2. Look-ahead Bias（未来情報リーク）
```
問題: horizon日後のy_retを特徴量計算時に参照してはいけない
確認: 特徴量はtまでの情報のみ使用、y_retはt+horizon
```

**検証方法**:
```python
# 特徴量の最終日時 < y_retの日時 を確認
assert features['date'].max() < labels['date'].max()
```

---

### 3. Walk-forward期間の調整

現在の設定:
```python
train_days = 180
val_days = 90
test_days = 60
step = 60
```

horizon=5の場合:
- test_daysの最後5日はy_retが欠損
- 実質test_days = 55日に減少

**対策**: test_days=65に増やす（5日のバッファ）

---

### 4. バックテストロジックの調整

現在: 毎日リバランス  
horizon=3: 3日ごとリバランスに変更が必要

```python
# src/model1/backtest.py 修正が必要

def run_backtest(
    predictions: pd.DataFrame,
    topk: int = 5,
    rebalance_days: int = 1,  # NEW: horizon期間
):
    # リバランスタイミングを制御
    pass
```

---

## 📊 期待効果の試算

### シナリオA: horizon=3

| 指標 | 現状 (h=1) | 期待 (h=3) | 改善度 |
|------|-----------|-----------|--------|
| Test IC | 0.0135 | 0.025 | +85% |
| Test Sharpe | 0.0026 | 0.04 | +1438% |
| MDD | -7.3% | -6.5% | +11% |
| 回転率 | 0.75 | 0.25 | -67% |
| 年間コスト | 15% | 5% | -67% |

---

### シナリオB: horizon=5

| 指標 | 現状 (h=1) | 期待 (h=5) | 改善度 |
|------|-----------|-----------|--------|
| Test IC | 0.0135 | 0.03 | +122% |
| Test Sharpe | 0.0026 | 0.06 | +2208% |
| MDD | -7.3% | -6.0% | +18% |
| 回転率 | 0.75 | 0.15 | -80% |
| 年間コスト | 15% | 3% | -80% |

---

## 🎯 成功基準

### Phase 1完了時
- ✅ horizon=3または5でIC > 0.02
- ✅ ICが現状(0.0135)より50%以上改善
- ✅ 実装にバグなし（look-ahead bias確認）

### Phase 2完了時
- ✅ 全fold平均IC >= 0.03
- ✅ Test Sharpe >= 0.05
- ✅ IC有意性 p < 0.05
- ✅ 良好fold >= 5/7

### Phase 3完了時
- ✅ Paper tradingで2週間プラス
- ✅ 取引コスト込みでSharpe > 0.03
- ✅ 実運用承認

---

## 📅 スケジュール

| Phase | 作業 | 期間 | 累積 |
|-------|------|------|------|
| Step 1 | コード実装 | 1日 | 1日 |
| Step 2 | 実験実行(h=1,3,5,10) | 2日 | 3日 |
| Step 3 | 結果分析・最適horizon決定 | 1日 | 4日 |
| Phase 2 | 全fold学習 | 2日 | 6日 |
| Phase 2 | 評価・調整 | 1日 | 7日 |
| Phase 3 | バックテスト調整 | 2日 | 9日 |
| Phase 3 | Paper trading | 14日 | 23日 |

**合計**: 約3週間（実装1週間 + Paper trading 2週間）

---

## 🛠 次のアクション（優先順位順）

### 🔥 今すぐ実施
1. **コード実装**（本日中）
   - Settings.label_horizon追加
   - _make_daily_labels_from_hourly修正
   - FeatureConfig更新

2. **horizon=3実験**（明日）
   - 特徴量再生成
   - 1 fold学習
   - IC確認

### 🔶 短期（2-3日以内）
3. **horizon=5実験**
4. **最適horizon決定**
5. **全fold学習**

### 🔷 中期（1週間以内）
6. **バックテストロジック調整**
7. **取引戦略最適化**

---

**最終更新**: 2026-02-10  
**次回レビュー**: 実験1完了後
