# ラベル期間延長機能 仕様書

**文書番号**: SPEC-001  
**バージョン**: 1.0  
**作成日**: 2026-02-10  
**作成者**: AI Assistant  
**承認者**: -  
**最終更新**: 2026-02-10

---

## 目次

1. [概要](#1-概要)
2. [背景と目的](#2-背景と目的)
3. [要件定義](#3-要件定義)
4. [設計](#4-設計)
5. [実装仕様](#5-実装仕様)
6. [テスト計画](#6-テスト計画)
7. [運用](#7-運用)
8. [付録](#8-付録)

---

## 1. 概要

### 1.1 機能概要

本機能は、株式リターン予測モデルのラベル（目的変数）として、従来の1日先リターンではなく、任意の期間（3日、5日、10日等）先のリターンを使用可能にするものである。

### 1.2 適用範囲

- **対象システム**: trading-bot（Model1）
- **対象モジュール**: `src/preprocessing/FeatureBuilder1h.py`
- **影響範囲**: 特徴量生成、モデル学習、バックテスト

### 1.3 用語定義

| 用語 | 定義 |
|------|------|
| **label_horizon** | ラベル期間。何営業日先のリターンを予測するか（単位: 営業日） |
| **IC** | Information Coefficient。予測値と実績値の相関係数 |
| **Sharpe Ratio** | リスク調整後リターン。(平均リターン - 無リスク金利) / リターンの標準偏差 |
| **Look-ahead Bias** | 未来情報リーク。特徴量計算時に未来の情報を使ってしまう誤り |

---

## 2. 背景と目的

### 2.1 現状の問題点

#### 2.1.1 予測精度の低さ
- **Test IC**: 0.0135（非常に低い）
- **統計的有意性**: p値 = 0.056（境界線）
- **原因**: 1日リターンは短期ノイズが大きく予測困難

#### 2.1.2 不安定性
- **Sharpe変動係数**: 45.18（非常に不安定）
- **Fold間のばらつき**: 7個中4個が損失
- **原因**: 短期市場ノイズの影響

#### 2.1.3 高い取引コスト
- **日次回転率**: 0.75
- **年間推定コスト**: 約15%
- **原因**: 毎日リバランスによる頻繁な売買

### 2.2 期待効果

#### 2.2.1 予測精度向上
- **IC目標**: 0.0135 → 0.03+（2倍以上）
- **メカニズム**: Signal-to-Noise比の改善

#### 2.2.2 安定性向上
- **Sharpe CV目標**: 45 → 5以下
- **メカニズム**: 短期変動の平準化

#### 2.2.3 コスト削減
- **回転率目標**: 0.75 → 0.3以下（60%削減）
- **メカニズム**: リバランス頻度の低下

### 2.3 定量的目標

| KPI | 現状 | 目標値 | 改善率 |
|-----|------|--------|--------|
| Test IC | 0.0135 | ≥ 0.030 | +122% |
| Test Sharpe | 0.0026 | ≥ 0.050 | +1823% |
| IC p値 | 0.056 | < 0.050 | - |
| Sharpe CV | 45.18 | < 5.0 | -89% |
| 良好Fold数 | 3/7 (43%) | ≥ 5/7 (71%) | +65% |
| 日次回転率 | 0.75 | ≤ 0.30 | -60% |
| Max Drawdown | -7.3% | ≤ -6.0% | +18% |

---

## 3. 要件定義

### 3.1 機能要件

#### FR-001: label_horizonパラメータの追加
- **優先度**: 必須（P0）
- **説明**: 環境変数`LABEL_HORIZON`でラベル期間を指定可能にする
- **デフォルト値**: 1（従来通り）
- **有効範囲**: 1 ～ 20（営業日）
- **検証**: 範囲外の値はエラーとする

#### FR-002: ラベル計算ロジックの変更
- **優先度**: 必須（P0）
- **説明**: `shift(-1)`を`shift(-label_horizon)`に変更
- **適用箇所**: 
  - `next_open`
  - `next_close`
  - `next_low`
  - `next_high`

#### FR-003: 複数のlabel_type対応
- **優先度**: 必須（P0）
- **説明**: 既存の3種類のlabel_typeで動作すること
  - `cc`: 引 → horizon日後の引
  - `oc`: horizon日後の寄 → 引
  - `oo`: 寄 → horizon日後の寄

#### FR-004: risk_label_typeの対応
- **優先度**: 必須（P0）
- **説明**: リスクラベルもhorizonに対応
  - `abs_ret`: horizon期間のリターンの絶対値
  - `drawdown`: horizon期間の最大ドローダウン
  - `intraday_range`: horizon日目の日中レンジ

### 3.2 非機能要件

#### NFR-001: 後方互換性
- **優先度**: 必須（P0）
- **説明**: `LABEL_HORIZON`未設定時は従来通り（horizon=1）で動作

#### NFR-002: Look-ahead Biasの防止
- **優先度**: 必須（P0）
- **説明**: 特徴量計算時刻 < ラベル計算時刻を保証
- **検証方法**: 自動テストで確認

#### NFR-003: データ欠損の明示
- **優先度**: 推奨（P1）
- **説明**: horizon日分のデータが末尾で欠損することをログ出力

#### NFR-004: 性能
- **優先度**: 推奨（P1）
- **説明**: 特徴量生成時間が従来比+10%以内
- **測定**: 100銘柄の処理時間

### 3.3 制約事項

#### C-001: データ長の制約
- horizon日分のデータが末尾で欠損
- 例: 709営業日 → horizon=5で704営業日に減少

#### C-002: Walk-forwardとの整合性
- test期間中のhorizon日は評価不可
- test_days設定を調整する必要がある

#### C-003: バックテストロジックの調整
- リバランス頻度をhorizon日ごとに変更する必要
- 現行システムでは毎日リバランス前提

---

## 4. 設計

### 4.1 システム構成

```
┌─────────────────────────────────────────────────────┐
│ Feature Generation (FeatureBuilder1h.py)            │
│  ┌────────────────────────────────────────────┐    │
│  │ Settings                                    │    │
│  │  - label_horizon: int (NEW)                │    │
│  └────────────────────────────────────────────┘    │
│                      ↓                              │
│  ┌────────────────────────────────────────────┐    │
│  │ FeatureConfig                               │    │
│  │  - label_horizon: int (NEW)                │    │
│  └────────────────────────────────────────────┘    │
│                      ↓                              │
│  ┌────────────────────────────────────────────┐    │
│  │ _make_daily_labels_from_hourly()           │    │
│  │  - shift(-label_horizon) (MODIFIED)        │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │ Features Parquet       │
         │  - y_ret (horizon期間) │
         │  - y_risk              │
         └────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │ Model Training         │
         └────────────────────────┘
```

### 4.2 データフロー

```
環境変数
  LABEL_HORIZON=3
       ↓
  Settings
       ↓
  FeatureConfig
       ↓
  build_features_for_ticker()
       ↓
  _make_daily_labels_from_hourly()
       ↓
  daily["next_close"] = daily["close_"].shift(-3)
       ↓
  daily["y_ret"] = log(next_close / close_)
       ↓
  Parquet出力
```

### 4.3 クラス図

```python
@dataclass
class Settings:
    prep_root: Path
    out_root: Path
    # ... 既存フィールド ...
    label_horizon: int  # NEW: デフォルト1

@dataclass
class FeatureConfig:
    tz: str
    windows: Tuple[int, ...]
    decision_hour: int
    label_type: str
    label_horizon: int  # NEW
    risk_label_type: str
    make_labels: bool

def _make_daily_labels_from_hourly(
    df_ohlcv: pd.DataFrame,
    label_type: str,
    label_horizon: int,  # NEW
    risk_label_type: str,
) -> pd.DataFrame:
    # 実装
```

### 4.4 シーケンス図

```
User -> FeatureBuilder1h: export LABEL_HORIZON=3
User -> FeatureBuilder1h: python FeatureBuilder1h.py
FeatureBuilder1h -> Settings: Load環境変数
Settings -> FeatureConfig: label_horizon=3
FeatureConfig -> build_features_for_ticker: 各銘柄処理
build_features_for_ticker -> _make_daily_labels_from_hourly: ラベル計算
_make_daily_labels_from_hourly -> DataFrame: shift(-3)適用
DataFrame -> ParquetFile: 特徴量保存
```

---

## 5. 実装仕様

### 5.1 修正箇所一覧

| No | ファイル | 関数/クラス | 行数 | 修正内容 | 優先度 |
|----|---------|------------|------|----------|--------|
| 1 | FeatureBuilder1h.py | Settings | ~24 | label_horizon追加 | P0 |
| 2 | FeatureBuilder1h.py | FeatureConfig | ~74 | label_horizon追加 | P0 |
| 3 | FeatureBuilder1h.py | _make_daily_labels_from_hourly | ~389 | 引数・ロジック修正 | P0 |
| 4 | FeatureBuilder1h.py | build_features_for_ticker | ~460 | 引数追加 | P0 |
| 5 | FeatureBuilder1h.py | run_all | ~650 | 引数追加 | P0 |

### 5.2 詳細実装

#### 5.2.1 Settings（修正1）

**場所**: `src/preprocessing/FeatureBuilder1h.py` line 24付近

**変更内容**:
```python
@dataclass
class Settings:
    # ... 既存フィールド ...
    
    # NEW: ラベル期間（営業日ベース）
    label_horizon: int = int(os.environ.get("LABEL_HORIZON", "1"))
```

**検証**:
```python
# 範囲チェック追加（推奨）
def __post_init__(self):
    if not (1 <= self.label_horizon <= 20):
        raise ValueError(f"label_horizon must be 1-20, got {self.label_horizon}")
```

---

#### 5.2.2 FeatureConfig（修正2）

**場所**: `src/preprocessing/FeatureBuilder1h.py` line 74付近

**変更内容**:
```python
@dataclass
class FeatureConfig:
    tz: str
    windows: Tuple[int, ...]
    decision_hour: int
    add_calendar: bool = True
    add_market: bool = True
    add_relative: bool = True
    make_labels: bool = True
    label_type: str = "cc"
    label_horizon: int = 1  # NEW
    risk_label_type: str = "drawdown"
```

---

#### 5.2.3 _make_daily_labels_from_hourly（修正3）

**場所**: `src/preprocessing/FeatureBuilder1h.py` line 389付近

**変更前**:
```python
def _make_daily_labels_from_hourly(
    df_ohlcv: pd.DataFrame,
    label_type: str = "oc",
    risk_label_type: str = "drawdown",
) -> pd.DataFrame:
    # ...
    daily["next_open"] = daily["open_"].shift(-1)
    daily["next_close"] = daily["close_"].shift(-1)
    daily["next_low"] = daily["low_"].shift(-1)
    daily["next_high"] = daily["high_"].shift(-1)
    # ...
```

**変更後**:
```python
def _make_daily_labels_from_hourly(
    df_ohlcv: pd.DataFrame,
    label_type: str = "oc",
    label_horizon: int = 1,  # NEW
    risk_label_type: str = "drawdown",
) -> pd.DataFrame:
    """
    1時間足データから日次ラベルを生成
    
    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        1時間足OHLCV
    label_type : str
        "oc", "cc", "oo"
    label_horizon : int
        何営業日先のリターンを計算するか（デフォルト: 1）
    risk_label_type : str
        "drawdown", "abs_ret", "intraday_range"
    
    Returns
    -------
    pd.DataFrame
        カラム: ["y_ret", "y_risk"]
    """
    df = df_ohlcv.copy()
    df["date"] = df.index.date

    daily = df.groupby("date").agg(
        open_=("Open", "first"),
        close_=("Close", "last"),
        low_=("Low", "min"),
        high_=("High", "max"),
    )

    # MODIFIED: shift(-1) → shift(-label_horizon)
    daily["next_open"] = daily["open_"].shift(-label_horizon)
    daily["next_close"] = daily["close_"].shift(-label_horizon)
    daily["next_low"] = daily["low_"].shift(-label_horizon)
    daily["next_high"] = daily["high_"].shift(-label_horizon)

    # 以降は従来通り（label_type分岐）
    if label_type == "oc":
        daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["next_open"] + EPS))
    elif label_type == "cc":
        daily["y_ret"] = np.log((daily["next_close"] + EPS) / (daily["close_"] + EPS))
    elif label_type == "oo":
        daily["y_ret"] = np.log((daily["next_open"] + EPS) / (daily["open_"] + EPS))
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    # risk_label_type処理（従来通り）
    if risk_label_type == "drawdown":
        if label_type == "oc":
            base = daily["next_open"]
        elif label_type == "cc":
            base = daily["close_"]
        else:
            base = daily["open_"]
        daily["y_risk"] = np.maximum(0.0, (base - daily["next_low"]) / (base + EPS))
    elif risk_label_type == "abs_ret":
        daily["y_risk"] = daily["y_ret"].abs()
    elif risk_label_type == "intraday_range":
        daily["y_risk"] = (daily["next_high"] - daily["next_low"]) / (daily["next_open"] + EPS)
        daily["y_risk"] = daily["y_risk"].clip(lower=0.0)
    else:
        raise ValueError(f"Unknown risk_label_type: {risk_label_type}")

    daily["y_risk"] = daily["y_risk"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    return daily[["y_ret", "y_risk"]]
```

---

#### 5.2.4 build_features_for_ticker（修正4）

**場所**: `src/preprocessing/FeatureBuilder1h.py` line 460付近

**変更前**:
```python
if cfg.make_labels:
    labels_df = _make_daily_labels_from_hourly(
        df_prep,
        label_type=cfg.label_type,
        risk_label_type=cfg.risk_label_type,
    )
```

**変更後**:
```python
if cfg.make_labels:
    labels_df = _make_daily_labels_from_hourly(
        df_prep,
        label_type=cfg.label_type,
        label_horizon=cfg.label_horizon,  # NEW
        risk_label_type=cfg.risk_label_type,
    )
```

---

#### 5.2.5 run_all（修正5）

**場所**: `src/preprocessing/FeatureBuilder1h.py` line 650付近

**変更前**:
```python
cfg = FeatureConfig(
    tz=settings.tz,
    windows=settings.windows,
    decision_hour=settings.decision_hour,
    label_type=settings.label_type,
    risk_label_type=settings.risk_label_type,
    make_labels=settings.make_labels,
)
```

**変更後**:
```python
cfg = FeatureConfig(
    tz=settings.tz,
    windows=settings.windows,
    decision_hour=settings.decision_hour,
    label_type=settings.label_type,
    label_horizon=settings.label_horizon,  # NEW
    risk_label_type=settings.risk_label_type,
    make_labels=settings.make_labels,
)
```

---

### 5.3 環境変数仕様

| 環境変数名 | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| LABEL_HORIZON | int | 1 | ラベル期間（営業日） | 3, 5, 10 |
| LABEL_TYPE | str | cc | ラベルタイプ | "cc", "oc", "oo" |
| RISK_LABEL_TYPE | str | drawdown | リスクラベルタイプ | "abs_ret" |

**使用例**:
```bash
export LABEL_HORIZON=5
export LABEL_TYPE=cc
export RISK_LABEL_TYPE=abs_ret
python src/preprocessing/FeatureBuilder1h.py
```

---

## 6. テスト計画

### 6.1 単体テスト

#### UT-001: Settings.label_horizon読み込み
```python
def test_settings_label_horizon_default():
    """デフォルト値が1であることを確認"""
    settings = Settings()
    assert settings.label_horizon == 1

def test_settings_label_horizon_from_env():
    """環境変数から読み込めることを確認"""
    os.environ["LABEL_HORIZON"] = "5"
    settings = Settings()
    assert settings.label_horizon == 5
```

#### UT-002: _make_daily_labels_from_hourly
```python
def test_label_horizon_1():
    """horizon=1で従来通りの動作"""
    df = create_sample_ohlcv()
    result = _make_daily_labels_from_hourly(df, label_type="cc", label_horizon=1)
    # 検証...

def test_label_horizon_5():
    """horizon=5で5日先のリターンを計算"""
    df = create_sample_ohlcv()
    result = _make_daily_labels_from_hourly(df, label_type="cc", label_horizon=5)
    # 検証...

def test_no_lookahead_bias():
    """Look-ahead biasがないことを確認"""
    # 特徴量日時 < ラベル日時 を検証
```

### 6.2 統合テスト

#### IT-001: エンドツーエンド
```bash
# horizon=3で特徴量生成→学習→評価
export LABEL_HORIZON=3
python src/preprocessing/FeatureBuilder1h.py
python src/model1/train_model1.py --max_folds 1
python evaluate_new_model.py
```

**期待結果**:
- エラーなく完了
- IC > 0.020
- 特徴量parquetにy_retが正しく格納

#### IT-002: 後方互換性
```bash
# LABEL_HORIZON未設定で従来通り動作
unset LABEL_HORIZON
python src/preprocessing/FeatureBuilder1h.py
# エラーなし、horizon=1相当
```

### 6.3 性能テスト

#### PT-001: 処理時間
```bash
# 100銘柄の処理時間測定
time python src/preprocessing/FeatureBuilder1h.py
```

**合格基準**: 従来比+10%以内

### 6.4 実験計画（詳細）

| 実験ID | Horizon | Max Folds | 目的 | 期待IC | 所要時間 |
|--------|---------|-----------|------|--------|----------|
| EXP-000 | 1 | 1 | ベースライン確認 | 0.0135 | 30分 |
| EXP-001 | 3 | 1 | 短期延長効果 | 0.020-0.025 | 1時間 |
| EXP-002 | 5 | 1 | 中期延長効果 | 0.025-0.035 | 1時間 |
| EXP-003 | 10 | 1 | 上限確認 | 0.015-0.025 | 1時間 |
| EXP-004 | 最良 | 7 | 全fold検証 | >= 0.030 | 7時間 |

**実験手順書**:
```bash
#!/bin/bash
# experiments/label_horizon_experiments.sh

for horizon in 1 3 5 10; do
    echo "=== Experiment: LABEL_HORIZON=$horizon ==="
    
    export LABEL_HORIZON=$horizon
    export LABEL_TYPE=cc
    
    # 特徴量生成
    python src/preprocessing/FeatureBuilder1h.py
    
    # 1 fold学習
    python src/model1/train_model1.py --max_folds 1
    
    # 結果保存
    cp -r artifacts/model1 artifacts/model1_exp_h${horizon}
    
    # 評価
    python evaluate_new_model.py > results/exp_h${horizon}_result.txt
    
    echo "Done: horizon=$horizon"
    echo ""
done

# 結果比較
python compare_experiments.py
```

---

## 7. 運用

### 7.1 デプロイ手順

#### Step 1: コード修正
```bash
# ブランチ作成
git checkout -b feature/label-horizon

# 修正実施（5箇所）
vim src/preprocessing/FeatureBuilder1h.py

# コミット
git add src/preprocessing/FeatureBuilder1h.py
git commit -m "feat: Add label_horizon parameter"
```

#### Step 2: 単体テスト
```bash
pytest tests/test_feature_builder.py -v
```

#### Step 3: 実験実行
```bash
bash experiments/label_horizon_experiments.sh
```

#### Step 4: 結果評価
```bash
python compare_experiments.py
cat results/experiment_summary.md
```

#### Step 5: マージ
```bash
# レビュー後
git checkout main
git merge feature/label-horizon
```

### 7.2 ロールバック手順

```bash
# 問題発生時
git revert <commit-hash>

# または特徴量再生成
export LABEL_HORIZON=1
python src/preprocessing/FeatureBuilder1h.py
```

### 7.3 モニタリング

#### 監視項目
- 特徴量生成時間
- y_retの欠損数
- IC推移
- Sharpe推移

#### アラート条件
- IC < 0.015（2週連続）
- Sharpe < 0（2週連続）
- MDD < -10%

### 7.4 メンテナンス

#### 定期レビュー
- **頻度**: 月次
- **内容**:
  - 最適horizonの再評価
  - 市場環境変化の確認
  - パフォーマンス劣化の検出

---

## 8. 付録

### 8.1 理論的背景

#### Signal-to-Noise比の改善

短期リターン（1日）:
```
R_1day = signal + noise
SNR = σ_signal / σ_noise
```

中期リターン（N日）:
```
R_Nday = N × signal + √N × noise
SNR_Nday = N × σ_signal / (√N × σ_noise) = √N × SNR_1day
```

**結論**: horizonをN倍にすると、SNRは√N倍に改善

**例**:
- horizon=1 → SNR = 1.0
- horizon=5 → SNR = √5 = 2.24（2.24倍改善）
- horizon=10 → SNR = √10 = 3.16（3.16倍改善）

---

### 8.2 期待効果の試算

#### シナリオA: horizon=3

| 指標 | 現状 (h=1) | 予測 (h=3) | 根拠 |
|------|-----------|-----------|------|
| IC | 0.0135 | 0.025 | SNR √3倍 → IC 1.7倍 |
| Sharpe | 0.0026 | 0.040 | IC改善 + ノイズ削減 |
| CV(Sharpe) | 45.18 | 8.0 | √3倍の安定化 |
| 回転率 | 0.75 | 0.25 | 3日ごとリバランス |
| コスト | 15% | 5% | 回転率に比例 |

#### シナリオB: horizon=5

| 指標 | 現状 (h=1) | 予測 (h=5) | 根拠 |
|------|-----------|-----------|------|
| IC | 0.0135 | 0.030 | SNR √5倍 → IC 2.2倍 |
| Sharpe | 0.0026 | 0.060 | IC改善 + ノイズ削減 |
| CV(Sharpe) | 45.18 | 6.0 | √5倍の安定化 |
| 回転率 | 0.75 | 0.15 | 5日ごとリバランス |
| コスト | 15% | 3% | 回転率に比例 |

---

### 8.3 リスク分析

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| ICが改善しない | 高 | 中 | 複数horizonを試す |
| 長すぎて予測困難 | 中 | 低 | horizon <= 10に制限 |
| Look-ahead bias | 高 | 低 | 自動テストで検証 |
| データ欠損増加 | 低 | 高 | ログで明示 |
| バックテスト調整漏れ | 中 | 中 | フェーズ分け実施 |

---

### 8.4 参考文献

1. **Quantitative Trading** - Ernest P. Chan (2009)
   - Chapter 3: Mean Reversion Strategies
   
2. **Advances in Financial Machine Learning** - Marcos Lopez de Prado (2018)
   - Chapter 3: Labeling
   
3. **Machine Learning for Asset Managers** - Marcos Lopez de Prado (2020)
   - Meta-Labeling

---

### 8.5 改訂履歴

| バージョン | 日付 | 著者 | 変更内容 |
|-----------|------|------|----------|
| 1.0 | 2026-02-10 | AI Assistant | 初版作成 |

---

### 8.6 承認

| 役割 | 氏名 | 日付 | 署名 |
|------|------|------|------|
| 作成者 | AI Assistant | 2026-02-10 | - |
| レビュアー | - | - | - |
| 承認者 | - | - | - |

---

**文書終了**
