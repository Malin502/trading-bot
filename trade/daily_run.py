# scripts/daily_run_v2.py
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


# ============================
# Settings (WRITE HERE)
# ============================

@dataclass
class DailyRunConfig:
    # --- どこをプロジェクトルートとみなすか（通常はこのままでOK） ---
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # --- 実行順 ---
    run_data_fetch: bool = True          # データ取得
    run_preprocessing: bool = True       # 前処理
    run_feature_update: bool = True      # 特徴量更新
    run_prediction: bool = True          # 推論

    # --- データ取得 ---
    data_fetch_cmd: list[str] = field(default_factory=lambda: ["python", "src/market_data/FetchTopix100_1h.py"])

    # --- 前処理 ---
    preprocessing_cmd: list[str] = field(default_factory=lambda: ["python", "src/preprocessing/PreprocessingIntraday1h.py"])

    # --- 特徴量更新 ---
    feature_update_cmd: list[str] = field(default_factory=lambda: ["python", "trade/update_features_model1.py"])

    # --- 推論 ---
    predict_cmd: list[str] = field(default_factory=lambda: ["python", "trade/predict_model1_latest.py"])


CFG = DailyRunConfig()


# ============================
# Helpers
# ============================

def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    """
    コマンドを実行し、stdout/stderrをコンソールに直接出力する。
    エラーなら例外を投げて即停止（中途半端な状態で推論しないため）。
    """
    env = os.environ.copy()
    pythonpath = str(cwd)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    print(f"[CMD] {' '.join(cmd)}")
    print(f"[CWD] {cwd}")
    print("=" * 80)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        env=env,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (code={proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    project_root = CFG.project_root

    ts = _now_str()
    print(f"[daily_run_v2] start {ts}")
    print(f"[daily_run_v2] project_root: {project_root}")
    print()

    # --- Step 1: data fetch ---
    if CFG.run_data_fetch:
        if not CFG.data_fetch_cmd:
            raise SystemExit("data_fetch_cmd is not set.")

        print("\n" + "=" * 80)
        print("[daily_run_v2] Step 1: Data Fetch")
        print("=" * 80)
        run_cmd(CFG.data_fetch_cmd, cwd=project_root)
        print("\n[daily_run_v2] ✓ data_fetch done")

    # --- Step 2: preprocessing ---
    if CFG.run_preprocessing:
        if not CFG.preprocessing_cmd:
            raise SystemExit("preprocessing_cmd is not set.")

        print("\n" + "=" * 80)
        print("[daily_run_v2] Step 2: Preprocessing")
        print("=" * 80)
        run_cmd(CFG.preprocessing_cmd, cwd=project_root)
        print("\n[daily_run_v2] ✓ preprocessing done")

    # --- Step 3: feature update ---
    if CFG.run_feature_update:
        if not CFG.feature_update_cmd:
            raise SystemExit(
                "feature_update_cmd is not set.\n"
                "Set CFG.feature_update_cmd to your feature update script command.\n"
                "Example:\n"
                "  CFG.feature_update_cmd = ['python', 'trade/update_features_model1.py']"
            )

        print("\n" + "=" * 80)
        print("[daily_run_v2] Step 3: Feature Update")
        print("=" * 80)
        run_cmd(CFG.feature_update_cmd, cwd=project_root)
        print("\n[daily_run_v2] ✓ feature_update done")

    # --- Step 4: prediction (ranking output) ---
    if CFG.run_prediction:
        predict_cmd = CFG.predict_cmd or ["python", "trade/predict_model1_latest.py"]
        print("\n" + "=" * 80)
        print("[daily_run_v2] Step 4: Prediction")
        print("=" * 80)
        run_cmd(predict_cmd, cwd=project_root)
        print("\n[daily_run_v2] ✓ prediction done")

    print("\n" + "=" * 80)
    print(f"[daily_run_v2] 🎉 All steps completed! {ts}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[daily_run_v2] ERROR:", str(e), file=sys.stderr)
        sys.exit(1)
