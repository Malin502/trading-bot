from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model import Model1ResidualMLP
    from backtest import BacktestConfig, run_topk_daily_backtest
    from metrics import rank_ic
except ImportError:
    from src.model1.model import Model1ResidualMLP
    from src.model1.backtest import BacktestConfig, run_topk_daily_backtest
    from src.model1.metrics import rank_ic

@dataclass
class TrainCfg:
    width: int = 256
    depth: int = 4
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8192

    max_epochs: int = 50
    patience: int = 8
    grad_clip: float = 1.0

    # task mode: "single" (returnのみ) | "multi" (return+risk)
    task_mode: str = "multi"

    lambda_risk: float = 0.4
    lambda_rank: float = 0.2
    huber_delta: float = 1.0
    rank_pair_samples: int = 4096
    rank_tie_threshold: float = 1e-6
    rank_weight_power: float = 1.0

    # early stopping: validation backtest (net) の制約付き最大化
    earlystop_topk: int = 5
    earlystop_score_method: str = "score_adj"  # スコア方式: ret_only | simple | cost_aware | sharpe_adj | utility
    earlystop_risk_aversion: float = 1.0
    earlystop_cost_bps: float = 5.0
    earlystop_slippage_bps: float = 3.0
    earlystop_turnover_penalty_bps: float = 8.0
    earlystop_min_pf: float = 1.0
    earlystop_max_drawdown_abs: float = 0.15
    earlystop_violation_penalty: float = 5.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

@torch.no_grad()
def predict_on_loader(model: nn.Module, loader, device: torch.device):
    model.eval()
    pred_ret_all, pred_risk_all, y_all = [], [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pr, pk = model(X)
        pred_ret_all.append(pr.detach().cpu().numpy())
        pred_risk_all.append(pk.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    pred_ret = np.concatenate(pred_ret_all, axis=0)
    pred_risk = np.concatenate(pred_risk_all, axis=0)
    y = np.concatenate(y_all, axis=0)  # (N,2)
    return pred_ret, pred_risk, y


def _build_eval_df(meta: pd.DataFrame, pred_ret: np.ndarray, pred_risk: np.ndarray) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)
    if "datetime" not in out.columns and "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "datetime"})
    required = ["datetime", "ticker", "y_ret"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"meta missing columns for backtest eval: {missing}")
    if len(out) != len(pred_ret):
        raise ValueError(f"meta/pred length mismatch: meta={len(out)} pred={len(pred_ret)}")
    out["pred_ret"] = pred_ret
    out["pred_risk"] = pred_risk
    return out


def _val_objective_with_constraints(
    summary: Dict[str, Any], 
    val_rank_ic: float,
    cfg: TrainCfg
) -> tuple[float, bool]:
    sharpe = float(summary["sharpe_daily"])
    pf = float(summary["pf"])
    mdd = float(summary["max_drawdown"])

    pf_gap = max(0.0, cfg.earlystop_min_pf - pf)
    mdd_gap = max(0.0, (-mdd) - cfg.earlystop_max_drawdown_abs)
    passed = (pf_gap == 0.0) and (mdd_gap == 0.0)

    penalty = cfg.earlystop_violation_penalty * (pf_gap + mdd_gap)
    # RankICを組み込んで、ランク相関も考慮
    score = sharpe + 0.5 * val_rank_ic - penalty
    return score, passed


def _use_risk_task(cfg: TrainCfg) -> bool:
    mode = str(cfg.task_mode).strip().lower()
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unknown task_mode: {cfg.task_mode}")
    return mode == "multi"


def _pairwise_rank_loss(
    pred_ret: torch.Tensor,
    y_ret: torch.Tensor,
    pair_samples: int,
    tie_threshold: float,
    weight_power: float,
) -> torch.Tensor:
    """
    Weighted pairwise ranking loss:
      softplus(-sign(y_i-y_j) * (pred_i-pred_j))
    where sample weights are based on |y_ret|.
    """
    n = int(pred_ret.shape[0])
    if n < 2 or pair_samples <= 0:
        return pred_ret.new_zeros(())

    i = torch.randint(0, n, (pair_samples,), device=pred_ret.device)
    j = torch.randint(0, n, (pair_samples,), device=pred_ret.device)
    valid = i != j
    if not torch.any(valid):
        return pred_ret.new_zeros(())
    i = i[valid]
    j = j[valid]

    y_diff = y_ret[i] - y_ret[j]
    sign = torch.sign(y_diff)
    valid = torch.abs(y_diff) > tie_threshold
    if not torch.any(valid):
        return pred_ret.new_zeros(())

    i = i[valid]
    j = j[valid]
    sign = sign[valid]

    pred_diff = pred_ret[i] - pred_ret[j]
    per_pair = F.softplus(-sign * pred_diff)

    # |ret|が大きいサンプルの寄与を強める
    w_i = torch.abs(y_ret[i]).clamp_min(1e-8).pow(weight_power)
    w_j = torch.abs(y_ret[j]).clamp_min(1e-8).pow(weight_power)
    w = w_i * w_j
    w = w / w.mean().clamp_min(1e-8)
    return torch.mean(per_pair * w)

def train_model1_one_fold(
    fold_id: int,
    train_loader,
    val_loader,
    test_loader,
    feature_cols: list[str],
    scaler,  # StandardScaler (foldごと)
    artifact_root: str | Path,
    cfg: TrainCfg,
    # 任意：predをdatetime/tickerへ紐付けたい場合に渡す（val/testそれぞれのindex順で並ぶ前提）
    val_meta: pd.DataFrame | None = None,   # columns: ["datetime","ticker","y_ret","y_risk"] など
    test_meta: pd.DataFrame | None = None,
):
    device = torch.device(cfg.device)
    fold_dir = _ensure_dir(Path(artifact_root) / f"fold_{fold_id:03d}")

    # --- model ---
    in_dim = len(feature_cols)
    model = Model1ResidualMLP(in_dim=in_dim, width=cfg.width, depth=cfg.depth, dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)

    best_state = None
    best_val_score = -float("inf")
    best_val_constraints_passed = False
    bad = 0

    print(f"[Fold {fold_id:03d}] Training start: max_epochs={cfg.max_epochs}, patience={cfg.patience}")
    use_risk_task = _use_risk_task(cfg)
    if not use_risk_task:
        print(f"[Fold {fold_id:03d}] task_mode=single: risk loss disabled")

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []
        train_rank_losses = []
        train_risk_losses = []

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # (B,2)
            y_ret = y[:, 0]
            y_risk = y[:, 1]

            pr, pk = model(X)
            loss_ret = loss_fn(pr, y_ret)
            if use_risk_task:
                loss_risk = loss_fn(pk, y_risk)
            else:
                loss_risk = loss_ret.new_zeros(())
            loss_rank = _pairwise_rank_loss(
                pred_ret=pr,
                y_ret=y_ret,
                pair_samples=cfg.rank_pair_samples,
                tie_threshold=cfg.rank_tie_threshold,
                weight_power=cfg.rank_weight_power,
            )
            loss = loss_ret + cfg.lambda_rank * loss_rank
            if use_risk_task:
                loss = loss + cfg.lambda_risk * loss_risk

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            train_losses.append(float(loss.detach().cpu().item()))
            train_rank_losses.append(float(loss_rank.detach().cpu().item()))
            train_risk_losses.append(float(loss_risk.detach().cpu().item()))

        # --- validation (early-stop: cost-adjusted backtest metrics with constraints) ---
        pr_val, pk_val, y_val = predict_on_loader(model, val_loader, device)
        val_bt_summary = None
        val_rank_ic_value = 0.0
        if val_meta is not None:
            val_eval_df = _build_eval_df(val_meta, pr_val, pk_val)
            val_bt_cfg = BacktestConfig(
                topk=cfg.earlystop_topk,
                score_method=cfg.earlystop_score_method,
                risk_aversion=cfg.earlystop_risk_aversion,
                cost_bps=cfg.earlystop_cost_bps,
                slippage_bps=cfg.earlystop_slippage_bps,
                turnover_penalty_bps=cfg.earlystop_turnover_penalty_bps,
            )
            val_bt_summary = run_topk_daily_backtest(
                val_eval_df,
                cfg=val_bt_cfg,
                datetime_col="datetime",
                ticker_col="ticker",
            )["summary"]
            # RankICを計算
            val_rank_ic_value = rank_ic(pr_val, y_val[:, 0])
            val_score, constraints_passed = _val_objective_with_constraints(val_bt_summary, val_rank_ic_value, cfg)
        else:
            # fallback（metaがない場合）
            yret_val = y_val[:, 0]
            yrisk_val = y_val[:, 1]
            val_rank_ic_value = rank_ic(pr_val, yret_val)
            mae_proxy = float(np.mean(np.abs(pr_val - yret_val)))
            if use_risk_task:
                mae_proxy += float(cfg.lambda_risk * np.mean(np.abs(pk_val - yrisk_val)))
            val_score = -mae_proxy + 0.5 * val_rank_ic_value
            constraints_passed = True

        # 進捗表示
        avg_train_loss = float(np.mean(train_losses))
        avg_rank_loss = float(np.mean(train_rank_losses)) if len(train_rank_losses) else 0.0
        avg_risk_loss = float(np.mean(train_risk_losses)) if len(train_risk_losses) else 0.0
        improved = "✓" if val_score > best_val_score else ""
        if val_bt_summary is not None:
            print(
                f"  Epoch {epoch+1:3d}/{cfg.max_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"RankLoss: {avg_rank_loss:.6f} | "
                f"RiskLoss: {avg_risk_loss:.6f} | "
                f"ValScore: {val_score:.6f} | "
                f"RankIC: {val_rank_ic_value:.4f} | "
                f"Sharpe(net): {val_bt_summary['sharpe_daily']:.4f} | "
                f"PF(net): {val_bt_summary['pf']:.4f} | "
                f"MDD(net): {val_bt_summary['max_drawdown']:.4f} | "
                f"Constraints: {'OK' if constraints_passed else 'NG'} | "
                f"Best: {best_val_score:.6f} {improved} | "
                f"Bad: {bad}/{cfg.patience}"
            )
        else:
            print(
                f"  Epoch {epoch+1:3d}/{cfg.max_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"RankLoss: {avg_rank_loss:.6f} | "
                f"RiskLoss: {avg_risk_loss:.6f} | "
                f"ValScore: {val_score:.6f} | "
                f"RankIC: {val_rank_ic_value:.4f} | "
                f"Best: {best_val_score:.6f} {improved} | "
                f"Bad: {bad}/{cfg.patience}"
            )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_constraints_passed = constraints_passed
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # bestロード
    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"[Fold {fold_id:03d}] Training completed. Best val score: {best_val_score:.6f}")

    # --- predictions for val/test ---
    pr_val, pk_val, y_val = predict_on_loader(model, val_loader, device)
    pr_test, pk_test, y_test = predict_on_loader(model, test_loader, device)

    # --- save artifacts ---
    torch.save(model.state_dict(), fold_dir / "model.pt")
    joblib.dump(scaler, fold_dir / "scaler.pkl")
    _save_json(feature_cols, fold_dir / "feature_cols.json")

    val_bt_final = None
    test_bt_final = None
    if val_meta is not None:
        val_eval_df = _build_eval_df(val_meta, pr_val, pk_val)
        val_bt_cfg = BacktestConfig(
            topk=cfg.earlystop_topk,
            score_method=cfg.earlystop_score_method,
            risk_aversion=cfg.earlystop_risk_aversion,
            cost_bps=cfg.earlystop_cost_bps,
            slippage_bps=cfg.earlystop_slippage_bps,
            turnover_penalty_bps=cfg.earlystop_turnover_penalty_bps,
        )
        val_bt_final = run_topk_daily_backtest(val_eval_df, cfg=val_bt_cfg, datetime_col="datetime", ticker_col="ticker")["summary"]
    if test_meta is not None:
        test_eval_df = _build_eval_df(test_meta, pr_test, pk_test)
        test_bt_cfg = BacktestConfig(
            topk=cfg.earlystop_topk,
            score_method=cfg.earlystop_score_method,
            risk_aversion=cfg.earlystop_risk_aversion,
            cost_bps=cfg.earlystop_cost_bps,
            slippage_bps=cfg.earlystop_slippage_bps,
            turnover_penalty_bps=cfg.earlystop_turnover_penalty_bps,
        )
        test_bt_final = run_topk_daily_backtest(test_eval_df, cfg=test_bt_cfg, datetime_col="datetime", ticker_col="ticker")["summary"]

    # metrics（最低限 + backtest系）
    metrics = {
        "fold": fold_id,
        "task_mode": cfg.task_mode,
        "best_val_score": best_val_score,
        "best_val_constraints_passed": bool(best_val_constraints_passed),
        # 互換キー（旧名）
        "best_val_proxy": best_val_score,
        "val": {
            "mae_ret": float(np.mean(np.abs(pr_val - y_val[:, 0]))),
            "mae_risk": float(np.mean(np.abs(pk_val - y_val[:, 1]))) if use_risk_task else float("nan"),
        },
        "test": {
            "mae_ret": float(np.mean(np.abs(pr_test - y_test[:, 0]))),
            "mae_risk": float(np.mean(np.abs(pk_test - y_test[:, 1]))) if use_risk_task else float("nan"),
        },
        "val_backtest": val_bt_final,
        "test_backtest": test_bt_final,
    }
    _save_json(metrics, fold_dir / "metrics.json")

    print(f"[Fold {fold_id:03d}] Final metrics:")
    print(f"  Val  - MAE Ret: {metrics['val']['mae_ret']:.6f}, MAE Risk: {metrics['val']['mae_risk']:.6f}")
    print(f"  Test - MAE Ret: {metrics['test']['mae_ret']:.6f}, MAE Risk: {metrics['test']['mae_risk']:.6f}")

    # preds保存（metaがある場合はdatetime/ticker付きで保存）
    if val_meta is not None:
        out = val_meta.copy()
        out["pred_ret"] = pr_val
        out["pred_risk"] = pk_val
        out.to_parquet(fold_dir / "preds_val.parquet", index=False)
    else:
        pd.DataFrame({
            "pred_ret": pr_val,
            "pred_risk": pk_val,
            "y_ret": y_val[:, 0],
            "y_risk": y_val[:, 1],
        }).to_parquet(fold_dir / "preds_val.parquet", index=False)

    if test_meta is not None:
        out = test_meta.copy()
        out["pred_ret"] = pr_test
        out["pred_risk"] = pk_test
        out.to_parquet(fold_dir / "preds_test.parquet", index=False)
    else:
        pd.DataFrame({
            "pred_ret": pr_test,
            "pred_risk": pk_test,
            "y_ret": y_test[:, 0],
            "y_risk": y_test[:, 1],
        }).to_parquet(fold_dir / "preds_test.parquet", index=False)

    return metrics
