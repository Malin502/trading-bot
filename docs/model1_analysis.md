# Model1 問題点分析レポート

**作成日**: 2026-02-09  
**対象**: trading-bot/artifacts/model1  
**モデル**: Model1ResidualMLP (315 features, 20 folds)

---

## 📊 現状の性能サマリー

### バックテスト結果（全20 fold平均）

| 指標 | Validation | Test | 差分 |
|------|-----------|------|------|
| **Sharpe Ratio** | 0.115 ± 0.096 | **-0.056 ± 0.202** | -0.171 |
| **IC (return)** | 0.027 ± 0.030 | **0.012 ± 0.075** | -0.015 |
| **Max Drawdown** | -6.4% ± 3.2% | N/A | - |
| **回転率** | 0.79 ± 0.06 | N/A | - |

### Fold別パフォーマンス（Test）

```
Fold  0: Sharpe= 0.0506, IC= 0.0140
Fold  1: Sharpe= 0.2467, IC= 0.0548
Fold  2: Sharpe=-0.0286, IC= 0.0765
Fold  3: Sharpe=-0.0264, IC=-0.0849
Fold  4: Sharpe=-0.4438, IC=-0.1497  ← 最悪
Fold  5: Sharpe= 0.1174, IC=-0.0046
Fold  6: Sharpe=-0.2930, IC= 0.0664
Fold  7: Sharpe=-0.2007, IC= 0.1537
Fold  8: Sharpe=-0.1625, IC= 0.0294
Fold  9: Sharpe=-0.2917, IC= 0.1412
```

**観察**: Fold間のばらつきが極めて大きい（Sharpe: -0.44 ～ +0.25）

---

## 🔴 重大な問題（Critical）

### 1. 予測性能の崩壊

#### A. 過学習
- **Validation → Test で性能が大幅劣化**
- Validation Sharpe 0.115 → Test Sharpe **-0.056**
- テストでは平均的にマイナスリターン

#### B. IC（情報係数）の低さ
```
Test IC (return): 0.012 ± 0.075
```
- 予測とリターンの相関がほぼゼロ
- **ランダム予測と同等レベル**
- 統計的有意性なし

#### C. 不安定性
- Fold間の標準偏差が平均値の3.6倍（Sharpe）
- 時期により性能が激変
- アンサンブルしても改善が限定的

**影響**: このモデルは実運用に使えない状態

---

### 2. リスク予測の機能不全

#### A. 全銘柄でリスクが均一
```python
# 推論結果例（2026-02-06）
7741.T: pred_risk = 0.276224
6701.T: pred_risk = 0.275141
6954.T: pred_risk = 0.275327
6098.T: pred_risk = 0.276317
...全銘柄が 0.275 ± 0.001
```

**問題**:
- 銘柄間のリスク差を学習できていない（差異 <1%）
- 高ボラティリティ銘柄と低ボラティリティ銘柄を区別不可
- **リスク調整したポートフォリオ構築が不可能**

#### B. モデル構造の問題
```python
# src/model1/model.py
self.head_risk = nn.Linear(width, 1)
pred_risk = F.softplus(self.head_risk(h).squeeze(-1))
```
- Softplus出力が特定値に収束
- 損失関数でリスク項が適切に最適化されていない可能性
- リスクラベル（y_risk）の品質問題の可能性

**影響**: マルチタスク学習の片方が完全に失敗

---

## 🟠 深刻な問題（Serious）

### 3. 過学習のリスク

#### A. モデルが複雑すぎる
```python
Model1ResidualMLP(
    in_dim=315,     # 特徴量数
    width=256,      # 隠れ層次元
    depth=4,        # Residual block数
    dropout=0.10
)
```
**推定パラメータ数**: 約1M（100万）個

**問題**:
- 学習データ: 100銘柄 × 限られた期間
- パラメータ/データ比が高すぎる
- Dropout 0.1では正則化不足

#### B. 特徴量が多すぎる
- 316カラム（numeric 314個）
- 重要な特徴が埋もれている
- 次元の呪い

**対策案**:
- モデルサイズ縮小: width=128, depth=2
- Dropout増加: 0.2-0.3
- 特徴量選択: 重要度上位50-100個

---

### 4. 特徴量の問題

#### A. 予測力不足
- IC 0.012 = ほぼゼロ
- Rolling window (8h, 24h, 56h) だけでは不十分
- セクター特徴、クロスセクショナル特徴も効いていない

#### B. 量vs質のトレードオフ
```
現在の特徴量構成:
- 基本特徴: logret, range, volume等
- Rolling特徴: 各窓幅で統計量
- テクニカル指標: RSI, BB, ATR, ADX, MACD, OBV
- 市場特徴: 市場インデックスとの相対
- セクター特徴: セクター平均との差分
- クロスセクショナル: rank, zscore
```
- すべて投入しても IC 0.012
- **特徴量追加では解決しない**

---

## 🟡 中程度の問題（Moderate）

### 5. 学習設定の問題

#### A. 時系列分割
- 20 fold（細かい）
- 各テスト期間: 20日と短い
- 短期ノイズの影響を受けやすい

#### B. ラベル定義
現在の設定:
```python
label_type: "oc"  # Open → Close
```
**確認が必要**:
- 未来情報リークの有無
- ラベル期間の妥当性
- リスクラベル（y_risk）の計算方法

---

### 6. 実運用上の問題

#### A. 高い回転率
```
平均日次回転率: 0.79
→ 年間約200回転
```
**影響**:
- 取引コスト大（8bps × 年200回 = 16%/年コスト）
- スリッページリスク
- 実運用では更に性能悪化

#### B. スコアマージンが小さい
```
トップ銘柄: score = 0.129 (12.9%)
取引コスト: 0.08% (8bps)
実質余地: 12.8%
```
- コスト後の利益余地が薄い
- 少しの予測誤差で赤字転落

#### C. 予測の不確実性
```python
# アンサンブル内のばらつき
pred_uncertainty = pred_ret_std / pred_ret.abs()
範囲: 0.5 ～ 11.2
```
- 多くの銘柄で予測が不安定
- 信頼できる予測が少ない

---

## 🟢 軽微な問題（Minor）

### 7. データ制約

#### A. 銘柄数
- Universe: 100銘柄（TOPIX構成銘柄の一部）
- クロスセクション学習には少ない

#### B. 学習期間
- **現状**: 709日（2023-02-24 ～ 2026-02-05）← 更新
- 新設定では5 foldのみ作成可能
- **統計的信頼性不足**: 10+ foldが望ましい
- 市場環境の変化への対応が不十分（約3年のみ）

---

## ✅ 優先順位付き対策案

### 🔥 最優先（すぐに実施）

#### 1. リスク予測の修正・簡略化
**オプションA**: リスク予測を廃止
```python
# Single-task learning
class Model1Simple(nn.Module):
    def forward(self, x):
        h = self.trunk(x)
        pred_ret = self.head_ret(h)
        return pred_ret  # リターンのみ
```
- マルチタスクの複雑さを排除
- リターン予測に集中

**オプションB**: リスク損失の重み調整
```python
loss = loss_ret + lambda_risk * loss_risk
# lambda_risk を 0.1 → 1.0 に増加
```

**オプションC**: リスクラベルの再定義
- 現在の計算方法を確認
- ボラティリティ、drawdownなど単純な指標に変更

---

#### 2. 過学習の抑制

**A. モデルサイズ削減**
```python
# Before
Model1ResidualMLP(in_dim=315, width=256, depth=4, dropout=0.10)

# After
Model1ResidualMLP(in_dim=100, width=128, depth=2, dropout=0.25)
```

**B. 特徴量選択**
```python
# 特徴量重要度計算 → 上位50-100個を選択
# 候補: SHAP, permutation importance
```

**C. 正則化強化**
- Dropout: 0.10 → 0.25
- Weight decay: 追加（1e-4）
- Early stopping: より厳格に

---

#### 3. ラベル定義の検証

**確認項目**:
```python
# 1. 未来情報リークチェック
# 2. ラベル期間の確認（O→C? C→C?）
# 3. リスクラベルの計算ロジック
# 4. データ前処理のタイミング
```

**実施**: コードレビュー → 必要に応じて修正

---

### 🔶 高優先（短期）

#### 4. シンプルなベースライン構築
```python
# Linear model / XGBoost でベースライン
# → ニューラルネットの必要性を検証
```

#### 5. IC向上施策
- より長期のラベル（1日 → 3-5日）でノイズ削減
- 銘柄選別（流動性低い銘柄を除外）
- 異常値処理の強化

#### 6. 汎化性能の改善
- Validation期間を延長（60日 → 120日）
- Test期間も延長（20日 → 60日）
- Fold数を削減（20 → 10）

---

### 🔷 中優先（中期）

#### 7. アンサンブル改善
- Fold選択の最適化（性能良いfoldのみ）
- ベイジアン最適化で重み付け

#### 8. データ拡充
- 銘柄数増加（100 → 全TOPIX）
- 学習期間延長

#### 9. 新特徴量の検討
- ファンダメンタル情報
- オルタナティブデータ
- マクロ経済指標

---

### 🔹 低優先（長期）

#### 10. アーキテクチャ改善
- Transformer系モデル
- Attention機構
- より高度な時系列モデル

#### 11. 取引コスト最適化
- 回転率削減施策
- 保有期間の延長

---

## 📈 推奨アクションプラン

### Phase 1: 診断（1週間）
1. ✅ 現状分析（完了）
2. ⬜ ラベル定義の検証
3. ⬜ データリークチェック
4. ⬜ 特徴量重要度分析

### Phase 2: クイックフィックス（2週間）
1. ⬜ シングルタスク化（リターンのみ）
2. ⬜ モデルサイズ削減
3. ⬜ 特徴量選択（上位100個）
4. ⬜ 正則化強化

### Phase 3: 再学習・評価（1週間）
1. ⬜ 新設定で全fold再学習
2. ⬜ バックテスト実施
3. ⬜ IC改善確認（目標: >0.05）
4. ⬜ Sharpe改善確認（目標: >0.2）

### Phase 4: 実運用準備（2週間）
1. ⬜ フォワードテスト（Paper trading）
2. ⬜ リスク管理ルール策定
3. ⬜ モニタリング体制構築

---

## 📝 メモ・その他

### データ品質
```
Features: (100, 316)
Date: 2026-02-09
Missing values: 0
Inf values: 0
```
✅ 特徴量自体の品質は問題なし

### 現在の推論設定
```python
N_RECENT_FOLDS = 5
ENSEMBLE_METHOD = "mean"
RISK_AVERSION = 0.5
UNCERTAINTY_PERCENTILE = 0.80
```

### 最新予測結果（2026-02-06）
```
Top ticker: 7741.T
- pred_ret: 16.76%（突出）
- pred_risk: 0.276（全銘柄同じ）
- score: 0.129
```
⚠️ 7741.Tの予測が楽観的すぎる可能性

---

## 🔗 関連ファイル

- モデル定義: `/workspaces/trading-bot/src/model1/model.py`
- 学習結果: `/workspaces/trading-bot/artifacts/model1/`
- 推論スクリプト: `/workspaces/trading-bot/trade/predict_model1_latest.py`
- 特徴量生成: `/workspaces/trading-bot/src/preprocessing/FeatureBuilder1h.py`
- 設定ファイル: `/workspaces/trading-bot/config/model1_train.yaml`

---

## 🔄 再評価結果（2026-02-10）

### Phase 2 改善実施後の状況

#### ✅ 実施済みの改善

1. **モデルサイズ削減**
   ```python
   # Before: width=256, depth=4, dropout=0.10
   # After:  width=128, depth=2, dropout=0.25
   ```

2. **正則化強化**
   - Dropout: 0.10 → 0.25
   - Weight decay: 1e-4 追加
   - Gradient clipping: 1.0

3. **損失関数の改善**
   ```python
   loss = loss_ret + lambda_risk * loss_risk + lambda_rank * loss_rank
   # lambda_risk = 0.2
   # lambda_rank = 0.2 (Pairwise ranking loss追加)
   ```

4. **Walk-forward設定変更**
   - Train: 240日
   - Val: 120日
   - Test: 20日 → **60日**（3倍に延長）
   - Step: 60日
   - **結果**: 1 fold = 420日 → 最大5 foldに削減（データ709日）

5. **Early Stopping改善**
   - Validation backtest metricベース
   - 制約条件追加（PF >= 1.0, MDD < 15%）
   - RankIC考慮

#### 📊 新モデルの性能（全5 fold完了）

**注**: 新設定ではデータ期間（709日）の制約から最大5 foldのみ作成可能。
artifacts/model1/のfold_005以降は旧設定（test_days=20）の残骸。

| 指標 | 旧モデル (20 fold) | 新モデル (5 fold) | 変化 |
|------|-------------------|------------------|------|
| **Test IC** | 0.012 ± 0.075 | **0.033 ± 0.045** | +175% ✅ |
| **Test Sharpe** | -0.056 ± 0.202 | **-0.043 ± 0.123** | +23% ⚠️ |
| **Test MDD** | N/A | -9.1% ± 3.8% | - |
| **Val IC** | 0.027 ± 0.030 | **0.018 ± 0.026** | -33% ⬇️ |
| **Val Sharpe** | 0.115 ± 0.096 | **0.084 ± 0.034** | -27% ⬇️ |

#### 🔍 詳細分析

##### 1. リスク予測の改善 ✅
```python
# 旧モデル（推論時）:
pred_risk: 0.275 ～ 0.277 (範囲: 0.002) ❌ 均一

# 新モデル（Fold 4 Test）:
pred_risk: 0.006 ～ 0.111 (範囲: 0.105) ✅ 差別化できている
Mean: 0.0091, Std: 0.0062
```
**評価**: リスク予測が機能し始めた

##### 2. リスクラベルの問題発見 ⚠️
```
y_risk 分布（全データ 70,234サンプル）:
- <0.01:  62.2%  ← 大半が小さい
- <0.05:  99.3%
- <0.10:  99.9%
- >=0.10: 0.1%

Mean: 0.010, Std: 0.010, Max: 0.171
```
**問題点**:
- ラベルが極端に小さい値に偏っている
- "リスク = Drawdown (寄/引 → 安値)" の定義が不適切
- ほとんどが1%未満のDD → 予測困難

**原因**:
```python
# FeatureBuilder1h.py
y_risk = max(0, (open - low) / open)  # label_type="oc"
```
- 日中の下落率を計算
- 上昇相場では0近傍に集中
- ボラティリティではなく下方リスクのみ

##### 3. Fold間のばらつき依然大きい
```
Fold 0: IC= 0.0085, Sharpe= 0.0798  ✅
Fold 1: IC= 0.1207, Sharpe=-0.1586  ⚠️
Fold 2: IC= 0.0131, Sharpe= 0.0668  ✅
Fold 3: IC= 0.0285, Sharpe= 0.0169  ⚠️
Fold 4: IC=-0.0063, Sharpe=-0.2195  ❌
```
- Sharpe: -0.22 ～ +0.08（レンジ0.30）
- IC: -0.006 ～ +0.121（レンジ0.13）
- 安定性に課題

**重要**: 新設定では**5 foldしか作れない**（データ709日 ÷ 420日/fold）
- サンプルサイズ不足で統計的信頼性が低い
- より長期データ（1000-1500日）の収集が急務

##### 4. 過学習は改善傾向
```
Val-Test Gap（旧モデル）:
- IC:     0.027 → 0.012 (差: -0.015)
- Sharpe: 0.115 → -0.056 (差: -0.171) ❌ 大きい

Val-Test Gap（新モデル）:
- IC:     0.018 → 0.033 (差: +0.015) ✅ Test良化
- Sharpe: 0.084 → -0.043 (差: -0.127) ⚠️ 依然ギャップあり
```
**評価**: ICは改善、Sharpeは依然として汎化に課題

#### 🎯 追加の問題点

##### A. リスクラベル定義の根本的欠陥
現在の定義（label_type="oc"）:
```python
y_risk = max(0, (next_open - next_low) / next_open)
```
**問題**:
1. 下方向のみ（上昇は0）→ 非対称
2. 日中変動のみ → オーバーナイトリスク無視
3. 実質ボラティリティではない

**改善案**:
```python
# 案1: 実現ボラティリティ
y_risk = rolling_std(returns, window=5)

# 案2: 絶対リターン
y_risk = abs(y_ret)

# 案3: 日中レンジ
y_risk = (high - low) / open
```

##### B. マルチタスク学習の難しさ
- リターン予測とリスク予測の目標が異なる
- リスクラベルの品質がリターン学習を阻害している可能性
- シングルタスク化の検討余地あり

#### 📋 更新された優先順位

### 🔥 最優先（即実施）

1. **リスクラベルの再定義** ← NEW
   ```python
   # Option 1: ボラティリティベース
   y_risk = rolling_std(logret, window=5)
   
   # Option 2: シンプルに絶対値
   y_risk = abs(y_ret)
   
   # Option 3: シングルタスク化
   # リスク予測を廃止してリターンのみ
   ```

2. **Walk-forward設定の調整** ← 修正
   - 現在: 5/5 fold完了（**全fold完了**）
   - データ: 709日（2023-02 ～ 2026-02）
   - 新設定では最大5 foldのみ作成可能
   - **代替案**: train/val/test期間を短縮して7-9 foldに増やす

3. **特徴量重要度分析**
   - 315特徴量のうち重要なものを特定
   - 不要特徴の除去（目標: 50-100個）

### 🔶 高優先

4. **ラベル期間の延長実験**
   - 現在: 1日リターン
   - 提案: 3-5日リターンでノイズ削減

5. **シンプルベースライン構築**
   - Linear Regression / Ridge
   - LightGBM
   - → NN の必要性検証

### 🔷 中優先

6. **データ期間の拡充** ← データ入手後に実施
   ```
   現状: データダウンロード制約により当面実施困難
   目標: 1200-1500日 → 10-15 fold（将来的に）
   ```

#### 💡 ポジティブな兆候

1. ✅ **Test ICが改善**: 0.012 → 0.033（+175%）
2. ✅ **リスク予測機能回復**: 差別化できるように
3. ✅ **Test期間延長**: より安定した評価
4. ✅ **過学習抑制**: Val-Test gapが縮小傾向

#### ⚠️ 依然として課題

1. ❌ **Test Sharpe依然マイナス**: -0.043（実運用不可）
2. ❌ **IC絶対値低い**: 0.033 < 0.05（目標未達）
3. ❌ **Fold間大きなばらつき**: 不安定
4. ❌ **リスクラベルの品質問題**: 根本的再設計必要
5. ❌ **サンプルサイズ不足**: 5 foldのみ（統計的信頼性低い）

---

## 🎯 現状優先すべき課題（2026-02-10時点）

### 🔥 最優先（P0: 即時対応）

#### 1. リスクラベルの再定義
**現状の問題**:
- y_risk分布が極端に偏っている（62% < 0.01）
- 予測困難なラベル → リスク予測が機能しづらい
- マルチタスク学習がリターン予測を阻害している可能性

**対策**:
```python
# 案A: シングルタスク化（推奨）
class Model1Simple(nn.Module):
    def forward(self, x):
        return self.head_ret(self.trunk(x))  # リターンのみ

# 案B: リスクラベルを実現ボラティリティに変更
y_risk = rolling_std(returns, window=5)

# 案C: 絶対リターンをリスクとする
y_risk = abs(y_ret)
```

**期待効果**:
- リターン予測の精度向上（IC: 0.033 → 0.05+）
- モデルの単純化による過学習抑制
- 学習の安定性向上

**工数**: 2-3日（実装 + 1 fold 検証）

---

#### 2. 特徴量の絞り込み
**現状の問題**:
- 315特徴量は多すぎる（過学習リスク）
- 不要特徴が多く含まれている可能性
- 重要特徴が埋もれている

**対策**:
1. Fold 0-4の学習済みモデルで特徴量重要度を計算
2. Permutation importance / SHAP values
3. 上位50-100特徴を選択
4. 選択後の特徴で1-2 fold再学習して検証

**期待効果**:
- 過学習抑制
- 学習時間短縮（50-70%削減）
- モデルの解釈性向上
- IC改善の可能性

**工数**: 3-5日（重要度計算 + 再学習 + 検証）

---

### 🔶 高優先（P1: 1-2週間以内）

#### 3. Walk-forward設定の調整（fold数増加）
**現状の問題**:
- 709日 → 最大5 foldのみ（現設定）
- 統計的信頼性不足（10+ fold望ましい）
- データ拡充は困難（ダウンロード制約）

**対策**: train/val/test期間を短縮してfold数を増やす
```python
# 現設定（5 fold）:
Train: 240日, Val: 120日, Test: 60日, Step: 60日
→ 1 fold = 420日 → 709日 ÷ 420日 = 5 fold

# 提案A（7 fold）:
Train: 180日, Val: 90日, Test: 60日, Step: 60日
→ 1 fold = 330日 → 709日 ÷ 330日 = 7 fold

# 提案B（9 fold）:
Train: 150日, Val: 75日, Test: 60日, Step: 45日
→ 1 fold = 285日 → 709日 ÷ 285日 = 9 fold
```

**期待効果**:
- Fold数増加 → 統計的信頼性向上
- データ収集不要 → 即実施可能
- より多様な時期で評価

**工数**: 1-2日（設定変更 + 再学習）

**注意点**:
- Train期間が短くなる → 学習データ減少
- 最適バランスの探索が必要

---

#### 4. データ拡充（データ収集後）← 優先度変更
**現状**: データダウンロード制約により当面実施困難

**対策**: データソースの確保後に実施
```
- 2021-2022年データ: 目標 1200日 → 10-12 fold
- 2019-2020年データ: 目標 1500日+ → 15+ fold
```

**工数**: 5-10日（データ入手後）

---

#### 5. シンプルベースラインの構築
**現状の問題**:
- ニューラルネットの必要性が不明
- IC 0.033は線形モデルで達成可能な可能性
- 複雑なモデルのコスパが悪い

**対策**:
```python
# 1. Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# 2. LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05
)

# 同じfold分割で評価・比較
```

**期待効果**:
- NN の相対的価値の把握
- より単純で解釈可能なモデルの可能性
- 開発・運用コスト削減

**工数**: 3-5日（実装 + 全fold評価）

---

#### 6. ラベル期間の延長実験
**現状の問題**:
- 1日リターンはノイズが多い
- 短期予測は困難
- 高回転率（0.79）→ コスト大

**対策**:
```python
# 現在: 1日リターン
y_ret = log(next_close / next_open)

# 提案: 3-5日リターン
y_ret_3d = log(close_t+3 / open_t+1)
y_ret_5d = log(close_t+5 / open_t+1)
```

**期待効果**:
- ノイズ削減 → IC向上
- 回転率低下 → コスト削減
- より安定した予測

**工数**: 2-3日（ラベル再生成 + 1 fold検証）

---

### 🔷 中優先（P2: 2-4週間以内）

#### 7. Early Stopping条件の見直し
**現状**:
- Validation backtestベース
- 制約: PF >= 1.0, MDD < 15%

**問題**:
- 制約が厳しすぎる可能性
- IC重視に変更すべきか

**対策**:
- IC最大化を主目標に変更
- Sharpe >= 0を制約条件に
- ハイパーパラメータチューニング

**期待効果**:
- より良いチェックポイント選択
- 汎化性能向上

**工数**: 2-3日

---

#### 8. アンサンブル手法の改善
**現状**:
- 全5 foldの単純平均
- 性能の悪いfoldも含まれる

**対策**:
```python
# 1. IC/Sharpeベースの重み付け
weights = [max(0, ic) for ic in fold_ics]
weights = weights / sum(weights)

# 2. 上位N foldのみ使用
top_n_folds = select_top_folds(folds, n=3, metric='ic')

# 3. Stacking
# Level 1: 各foldの予測
# Level 2: Meta-learner
```

**期待効果**:
- 予測精度向上
- 不安定なfoldの影響軽減

**工数**: 3-5日

---

#### 9. クロスセクション特徴の強化
**現状**:
- rank, zscore のみ
- 相対的な情報が不足

**対策**:
```python
# 銘柄間の相対性を強化
- セクター内順位
- 市場全体との偏差
- 過去N日の順位変動
- 出来高ランク
```

**期待効果**:
- クロスセクション alpha捕捉
- IC向上

**工数**: 3-4日

---

### 🔹 低優先（P3: 1-2ヶ月以内）

#### 10. モデルアーキテクチャの探索
- Transformer
- Attention機構
- GRU/LSTM

**前提**: ベースライン構築後、NN の価値が確認できた場合のみ

**工数**: 1-2週間

---

#### 11. 新データソースの追加
- ファンダメンタル情報
- センチメント分析
- マクロ経済指標

**工数**: 2-4週間

---

#### 12. 取引戦略の最適化
- ポジションサイジング
- リバランスタイミング
- ストップロス/テイクプロフィット

**工数**: 1-2週間

---

## 📅 推奨実施スケジュール

### Week 1-2: クイックウィン
```
Day 1-2:  特徴量重要度計算 → 上位100選択
Day 3-5:  シングルタスク化（リターンのみ）+ 1 fold検証
Day 6-7:  選択特徴 + シングルタスクで全fold再学習
Day 8-10: ベースライン構築（Ridge, LightGBM）
```
**目標**: IC 0.05+, Sharpe 0.0+ 達成

### Week 3-4: Walk-forward調整 + 精度向上
```
Day 1-2:  Walk-forward設定調整（7-9 fold化）
Day 3-5:  新設定で全fold再学習
Day 6-8:  ラベル期間延長実験（3日, 5日）
Day 9-10: アンサンブル改善
```
**目標**: 統計的信頼性確保（7+ fold）、IC改善

### Week 5-6: 最適化
```
Day 1-3:  Early stopping調整
Day 4-7:  クロスセクション特徴強化
Day 8-10: ハイパーパラメータチューニング
```
**目標**: IC 0.10+, Sharpe 0.2+

### Week 7-8: 実運用準備
```
Day 1-5:  Paper trading
Day 6-10: リスク管理・モニタリング体制構築
```

---

## 🎯 成功基準（KPI）

### Phase 1完了時（Week 2）
- ✅ Test IC >= 0.05
- ✅ Test Sharpe >= 0.0
- ✅ 特徴量数 <= 100
- ✅ シングルタスク化完了

### Phase 2完了時（Week 4）
- ✅ Fold数 >= 7
- ✅ Fold間Sharpe標準偏差 < 0.15
- ✅ ベースラインとの比較完了

### Phase 3完了時（Week 6）
- ✅ Test IC >= 0.08
- ✅ Test Sharpe >= 0.15
- ✅ Val-Test gap < 0.05（IC）

### 実運用判断基準（Week 8）
- ✅ Test IC >= 0.10
- ✅ Test Sharpe >= 0.20
- ✅ 2週間Paper trading でSharpe > 0
- ✅ Max Drawdown < 10%

---

## 🛠 追加実装（2026-02-10 追記）

### 1) リスクラベル再定義を実装
対象: `src/preprocessing/FeatureBuilder1h.py`

- `risk_label_type` を追加し、以下を切替可能化
  - `drawdown`（従来）
  - `abs_ret`
  - `intraday_range`
- デフォルトを `abs_ret` に変更
- 環境変数で上書き可能
  - `RISK_LABEL_TYPE=drawdown|abs_ret|intraday_range`

### 2) シングルタスク学習を実装
対象: `src/model1/_train.py`

- `task_mode` を追加（`single` / `multi`）
- `single` 時はリスク損失を無効化
- `mae_risk` は `single` 時に `NaN` として保存
- early stopping のデフォルトスコアを `ret_only` に変更

### 3) 特徴量選択（上位100）を実装
対象: `src/preprocessing/DatasetBuilder.py`

- foldごとに trainデータのみを使って特徴量選択（リーク回避）
- 指標: `|corr(feature, y_ret)|` の上位N
- 設定: `DataLoadersConfig.feature_top_n`
- 現在設定: `feature_top_n=100`

### 4) Walk-forward設定を7 fold化
対象: `src/model1/train_model1.py`, `config/model1_train.yaml`

- `train/val/test/step = 180/90/60/60`
- `min_unique_days = 330`
- 実データ（709営業日）で fold数は **7** を確認

### 5) 推論・バックテストのスコア方式拡張
対象: `trade/predict_model1_latest.py`, `src/model1/backtest.py`

- `ret_only` スコア方式を追加（リスク予測に依存しない）
- 学習の `task_mode=single` と整合するよう調整

### 6) 動作確認結果（実装直後）

- fold生成: `7` fold（確認済み）
- 特徴量数: `315 -> 100`（確認済み）
- 新規ラベル生成: `drawdown/abs_ret/intraday_range` の3方式で出力確認済み
- 注意: **全fold再学習・本評価は未実施**（次ステップ）

---

**最終更新**: 2026-02-10  
**次回レビュー予定**: Phase 1完了時（Week 2）
