# scripts/train_model1_with_existing_dataset.py
from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.DatasetBuilder import (
    DatasetSettings, DataLoadersConfig,
    load_universe_table, clean_table,
    WalkForwardConfig, make_walk_forward_folds,
    build_fold_datasets_and_loaders,
)

try:
    from _train import TrainCfg, train_model1_one_fold
except ImportError:
    # package import (e.g. `python -m src.model1.train_model1`)
    from src.model1._train import TrainCfg, train_model1_one_fold


def fold_provider_existing(
    s: DatasetSettings,
    wf: WalkForwardConfig,
    dl_cfg: DataLoadersConfig,
):
    df = load_universe_table(s)
    df, feature_cols = clean_table(df, s, feature_cols=None)

    folds = make_walk_forward_folds(df, wf)

    for fold_id, fold in enumerate(folds):
        pack = build_fold_datasets_and_loaders(df, feature_cols, fold, s, dl_cfg)
        if len(pack["feature_cols"]) != len(feature_cols):
            print(
                f"[Fold {fold_id:03d}] feature selection: "
                f"{len(feature_cols)} -> {len(pack['feature_cols'])}"
            )

        train_df = pack["train_df"]
        val_df = pack["val_df"]
        test_df = pack["test_df"]

        # 予測保存用メタ（Datasetと同順であることが前提：ここではdf抽出順＝Dataset生成順）
        # indexがDatetimeなので列に落とす
        val_meta = val_df.reset_index()[["Datetime", s.id_col, *s.y_cols]].rename(columns={"Datetime": "datetime"})
        test_meta = test_df.reset_index()[["Datetime", s.id_col, *s.y_cols]].rename(columns={"Datetime": "datetime"})

        # feature_cols の取り出し：パッチ適用後は pack["feature_cols"] がある
        # パッチ未適用なら pack["feature_cols_scaled"] しかないのでフォールバック
        feat_cols_used = pack.get("feature_cols", None)
        if feat_cols_used is None:
            feat_cols_used = pack["feature_cols_scaled"]  # ※この場合 sin/cos が入力から消える点は仕様ズレ

        yield {
            "fold_id": fold_id,
            "train_loader": pack["dl_train"],
            "val_loader": pack["dl_val"],
            "test_loader": pack["dl_test"],
            "scaler": pack["scaler"],
            "feature_cols": feat_cols_used,
            "val_meta": val_meta,
            "test_meta": test_meta,
        }

def main():
    s = DatasetSettings(
        universe_parquet=Path("features/universe/features_1h_for_model1_universe.parquet"),
        require_labels=True,
    )

    wf = WalkForwardConfig(
        train_days=180,
        val_days=90,
        test_days=60,
        step_days=60,
        min_unique_days=330,
    )

    dl_cfg = DataLoadersConfig(
        batch_size=512,
        num_workers=0,
        pin_memory=True,
        feature_top_n=0,
    )

    train_cfg = TrainCfg(
        width=128,
        depth=2,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        patience=12,
        task_mode="single",
        lambda_risk=0.0,
        earlystop_score_method="ret_only",
        huber_delta=1.0,
        grad_clip=1.0,
    )

    artifact_root = Path("artifacts/model1")
    artifact_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for pack in fold_provider_existing(s, wf, dl_cfg):
        print(f"\n{'='*80}")
        print(f"Starting Fold {pack['fold_id']}")
        print(f"{'='*80}")
        
        m = train_model1_one_fold(
            fold_id=pack["fold_id"],
            train_loader=pack["train_loader"],
            val_loader=pack["val_loader"],
            test_loader=pack["test_loader"],
            feature_cols=pack["feature_cols"],
            scaler=pack["scaler"],
            artifact_root=artifact_root,
            cfg=train_cfg,
            val_meta=pack["val_meta"],
            test_meta=pack["test_meta"],
        )
        all_metrics.append(m)
        
        print(f"\n{'='*80}")
        print(f"Fold {pack['fold_id']} Summary:")
        print(f"  Best Val Score: {m.get('best_val_score', m['best_val_proxy']):.6f}")
        print(f"  Constraints OK: {m.get('best_val_constraints_passed', False)}")
        print(f"  Val  - MAE Ret: {m['val']['mae_ret']:.6f}, MAE Risk: {m['val']['mae_risk']:.6f}")
        print(f"  Test - MAE Ret: {m['test']['mae_ret']:.6f}, MAE Risk: {m['test']['mae_risk']:.6f}")
        print(f"{'='*80}\n")

    (artifact_root / "all_folds_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    print(f"\n{'='*80}")
    print("All Folds Completed!")
    print(f"{'='*80}")
    print(f"Total folds: {len(all_metrics)}")
    
    # 全体の平均メトリクスを表示
    avg_val_ret = sum(m['val']['mae_ret'] for m in all_metrics) / len(all_metrics)
    avg_val_risk = float(np.nanmean([m['val']['mae_risk'] for m in all_metrics]))
    avg_test_ret = sum(m['test']['mae_ret'] for m in all_metrics) / len(all_metrics)
    avg_test_risk = float(np.nanmean([m['test']['mae_risk'] for m in all_metrics]))
    
    print(f"\nAverage Metrics across all folds:")
    print(f"  Val  - MAE Ret: {avg_val_ret:.6f}, MAE Risk: {avg_val_risk:.6f}")
    print(f"  Test - MAE Ret: {avg_test_ret:.6f}, MAE Risk: {avg_test_risk:.6f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
