# src/model1/runner.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Any, Optional, List

import pandas as pd

from _train import TrainCfg, train_model1_one_fold


FoldPack = Dict[str, Any]
FoldProvider = Callable[[], Iterator[FoldPack]]
# FoldProvider は「fold_packを順にyieldする関数」


def train_all_folds(
    fold_provider: FoldProvider,
    artifact_root: str | Path,
    cfg: TrainCfg,
) -> list[dict]:
    """
    fold_provider: 既存のデータセット生成コードをラップして
                   fold_pack(dict) をyieldする関数
    """
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for pack in fold_provider():
        fold_id: int = int(pack["fold_id"])

        metrics = train_model1_one_fold(
            fold_id=fold_id,
            train_loader=pack["train_loader"],
            val_loader=pack["val_loader"],
            test_loader=pack["test_loader"],
            feature_cols=pack["feature_cols"],
            scaler=pack["scaler"],
            artifact_root=artifact_root,
            cfg=cfg,
            val_meta=pack.get("val_meta"),
            test_meta=pack.get("test_meta"),
        )
        all_metrics.append(metrics)

    # 集約保存したい場合は呼び出し側でJSON保存すればOK
    return all_metrics
