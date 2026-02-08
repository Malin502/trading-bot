# src/model1/analyze_features.py
"""
特徴量分析：IC、相関、分布、重要度などを可視化
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================
# Settings
# ============================

UNIVERSE_PARQUET = Path("features/universe/features_1h_for_model1_universe.parquet")
OUTPUT_DIR = Path("artifacts/model1/feature_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = ["datetime", "ticker", "y_ret", "y_risk"]


# ============================
# Helpers
# ============================

def load_data(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """データ読み込みと特徴量列の推論"""
    df = pd.read_parquet(path)
    
    # DatetimeIndexをdatetime列に変換
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # 特徴量列の推論（required以外の数値列）
    feature_cols = []
    for c in df.columns:
        if c in REQUIRED_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    
    print(f"✓ Loaded: {len(df):,} rows × {len(df.columns)} cols")
    print(f"✓ Features: {len(feature_cols)} numeric columns")
    print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df, feature_cols


def analyze_basic_stats(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """基本統計量"""
    stats_data = []
    
    for col in feature_cols:
        s = df[col]
        stats_data.append({
            "feature": col,
            "count": int(s.count()),
            "missing_pct": float(s.isna().sum() / len(s) * 100),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "25%": float(s.quantile(0.25)),
            "50%": float(s.quantile(0.50)),
            "75%": float(s.quantile(0.75)),
            "max": float(s.max()),
            "skew": float(s.skew()),
            "kurt": float(s.kurtosis()),
        })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df


def calculate_ic(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Information Coefficient (IC)：各特徴量とy_retの相関
    日次で計算して平均を取る
    """
    df = df.dropna(subset=["y_ret"])
    
    ic_results = []
    
    for col in feature_cols:
        daily_ics = []
        daily_rankics = []
        
        for date, g in df.groupby(df["datetime"].dt.date):
            if len(g) < 5:  # 最低5銘柄
                continue
            
            x = g[col].values
            y = g["y_ret"].values
            
            # 欠損値処理
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Pearson IC
            if x_clean.std() > 0 and y_clean.std() > 0:
                ic = float(np.corrcoef(x_clean, y_clean)[0, 1])
                daily_ics.append(ic)
            
            # Spearman RankIC
            try:
                rankic, _ = stats.spearmanr(x_clean, y_clean)
                daily_rankics.append(float(rankic))
            except:
                pass
        
        # 集計
        if daily_ics:
            ic_results.append({
                "feature": col,
                "ic_mean": float(np.mean(daily_ics)),
                "ic_std": float(np.std(daily_ics)),
                "ic_ir": float(np.mean(daily_ics) / (np.std(daily_ics) + 1e-9)),
                "rankic_mean": float(np.mean(daily_rankics)) if daily_rankics else 0.0,
                "rankic_std": float(np.std(daily_rankics)) if daily_rankics else 0.0,
                "n_days": len(daily_ics),
            })
    
    ic_df = pd.DataFrame(ic_results).sort_values("ic_mean", ascending=False, key=abs)
    return ic_df


def plot_ic_chart(ic_df: pd.DataFrame, output_dir: Path):
    """IC分布の可視化"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # IC Mean
    top_n = 30
    ic_top = ic_df.head(top_n)
    
    ax = axes[0]
    colors = ['green' if x > 0 else 'red' for x in ic_top["ic_mean"]]
    ax.barh(range(len(ic_top)), ic_top["ic_mean"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(ic_top)))
    ax.set_yticklabels(ic_top["feature"], fontsize=8)
    ax.set_xlabel("IC Mean (Pearson)", fontsize=10)
    ax.set_title(f"Top {top_n} Features by IC Mean", fontsize=12, fontweight="bold")
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3)
    
    # RankIC Mean
    ic_top_rank = ic_df.sort_values("rankic_mean", ascending=False, key=abs).head(top_n)
    
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in ic_top_rank["rankic_mean"]]
    ax.barh(range(len(ic_top_rank)), ic_top_rank["rankic_mean"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(ic_top_rank)))
    ax.set_yticklabels(ic_top_rank["feature"], fontsize=8)
    ax.set_xlabel("RankIC Mean (Spearman)", fontsize=10)
    ax.set_title(f"Top {top_n} Features by RankIC Mean", fontsize=12, fontweight="bold")
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ic_ranking.png", dpi=120, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'ic_ranking.png'}")
    plt.close()


def calculate_feature_correlation(df: pd.DataFrame, feature_cols: List[str], top_n: int = 50) -> pd.DataFrame:
    """特徴量間の相関行列（多重共線性チェック）"""
    # サンプリング（計算高速化）
    if len(df) > 100000:
        df_sample = df.sample(n=100000, random_state=42)
    else:
        df_sample = df
    
    # 上位N個の特徴量のみ
    feature_subset = feature_cols[:top_n]
    
    corr_matrix = df_sample[feature_subset].corr()
    return corr_matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_dir: Path):
    """相関ヒートマップ"""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(
        corr_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    
    ax.set_title("Feature Correlation Matrix (Top 50)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=120, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'correlation_heatmap.png'}")
    plt.close()


def detect_multicollinearity(corr_matrix: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """高相関ペアの検出"""
    high_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": float(corr_val),
                })
    
    if not high_corr:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    return pd.DataFrame(high_corr).sort_values("correlation", ascending=False, key=abs)


def plot_target_distributions(df: pd.DataFrame, output_dir: Path):
    """目標変数の分布確認"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # y_ret: ヒストグラム
    ax = axes[0, 0]
    df["y_ret"].hist(bins=100, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title("y_ret Distribution", fontweight="bold")
    ax.set_xlabel("y_ret (log return)")
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    
    # y_ret: QQプロット
    ax = axes[0, 1]
    stats.probplot(df["y_ret"].dropna(), dist="norm", plot=ax)
    ax.set_title("y_ret Q-Q Plot", fontweight="bold")
    
    # y_ret: 時系列
    ax = axes[0, 2]
    daily_mean = df.groupby(df["datetime"].dt.date)["y_ret"].mean()
    daily_mean.plot(ax=ax, linewidth=0.8)
    ax.set_title("y_ret Daily Mean", fontweight="bold")
    ax.set_xlabel("Date")
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3)
    
    # y_risk: ヒストグラム
    ax = axes[1, 0]
    df["y_risk"].hist(bins=100, ax=ax, edgecolor='black', alpha=0.7, color='orange')
    ax.set_title("y_risk Distribution", fontweight="bold")
    ax.set_xlabel("y_risk")
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    
    # y_risk: QQプロット
    ax = axes[1, 1]
    stats.probplot(df["y_risk"].dropna(), dist="norm", plot=ax)
    ax.set_title("y_risk Q-Q Plot", fontweight="bold")
    
    # y_risk: 時系列
    ax = axes[1, 2]
    daily_mean_risk = df.groupby(df["datetime"].dt.date)["y_risk"].mean()
    daily_mean_risk.plot(ax=ax, linewidth=0.8, color='orange')
    ax.set_title("y_risk Daily Mean", fontweight="bold")
    ax.set_xlabel("Date")
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "target_distributions.png", dpi=120, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'target_distributions.png'}")
    plt.close()


def main():
    print("=" * 80)
    print("Feature Analysis Start")
    print("=" * 80)
    
    # データ読み込み
    df, feature_cols = load_data(UNIVERSE_PARQUET)
    
    # 1. 基本統計
    print("\n[1/6] Basic Statistics...")
    stats_df = analyze_basic_stats(df, feature_cols)
    stats_df.to_csv(OUTPUT_DIR / "basic_stats.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'basic_stats.csv'}")
    print(f"\n欠損値が多い特徴量（>10%）:")
    high_missing = stats_df[stats_df["missing_pct"] > 10].sort_values("missing_pct", ascending=False)
    if len(high_missing) > 0:
        print(high_missing[["feature", "missing_pct"]].to_string(index=False))
    else:
        print("  なし")
    
    # 2. IC分析（最重要）
    print("\n[2/6] Information Coefficient (IC) Analysis...")
    ic_df = calculate_ic(df, feature_cols)
    ic_df.to_csv(OUTPUT_DIR / "ic_analysis.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'ic_analysis.csv'}")
    print(f"\nTop 10 Features by |IC|:")
    print(ic_df.head(10)[["feature", "ic_mean", "ic_ir", "rankic_mean"]].to_string(index=False))
    print(f"\nIC統計:")
    print(f"  IC Mean (全体平均): {ic_df['ic_mean'].mean():.6f}")
    print(f"  |IC| > 0.02: {(ic_df['ic_mean'].abs() > 0.02).sum()} features")
    print(f"  |IC| > 0.05: {(ic_df['ic_mean'].abs() > 0.05).sum()} features")
    
    # 3. IC可視化
    print("\n[3/6] Plotting IC Rankings...")
    plot_ic_chart(ic_df, OUTPUT_DIR)
    
    # 4. 目標変数分布
    print("\n[4/6] Target Variable Distributions...")
    plot_target_distributions(df, OUTPUT_DIR)
    
    # 5. 特徴量間相関
    print("\n[5/6] Feature Correlation Analysis...")
    corr_matrix = calculate_feature_correlation(df, feature_cols, top_n=50)
    corr_matrix.to_csv(OUTPUT_DIR / "feature_correlation.csv")
    print(f"✓ Saved: {OUTPUT_DIR / 'feature_correlation.csv'}")
    plot_correlation_heatmap(corr_matrix, OUTPUT_DIR)
    
    # 6. 多重共線性チェック
    print("\n[6/6] Multicollinearity Detection (|corr| > 0.95)...")
    high_corr_df = detect_multicollinearity(corr_matrix)
    if len(high_corr_df) > 0:
        high_corr_df.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False)
        print(f"✓ Found {len(high_corr_df)} high-correlation pairs")
        print(high_corr_df.head(10).to_string(index=False))
    else:
        print("  高相関ペアなし")
    
    # サマリー
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"  Total Features: {len(feature_cols)}")
    print(f"  Features with |IC| > 0.02: {(ic_df['ic_mean'].abs() > 0.02).sum()}")
    print(f"  Features with |IC| > 0.05: {(ic_df['ic_mean'].abs() > 0.05).sum()}")
    print(f"  Average |IC|: {ic_df['ic_mean'].abs().mean():.6f}")
    print(f"  High missing (>10%): {(stats_df['missing_pct'] > 10).sum()}")
    print(f"  High correlation pairs: {len(high_corr_df)}")
    print(f"\n⚠️  IC < 0.02 は予測力が非常に弱い指標です。")
    print(f"    実用的なモデルには |IC| > 0.05 の特徴量が複数必要です。")
    print(f"\n📁 Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
