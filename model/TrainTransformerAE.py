from __future__ import annotations
import sys

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.intraday_1h_ae.datasets_ae_1h import WindowedFeaturesAEDataset
from model.TransformerAE import TransformerAE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_SAMPLES = PROJECT_ROOT / "datasets/intraday_1h_ae/samples.parquet"
CKPT_DIR = PROJECT_ROOT / "model/checkpoints/ae_1h"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 56
LATENT_DIM = 32

# AE学習期間（実運用想定：2026年は未知として残す）
TRAIN_START = "2023-02-01"
TRAIN_END = "2025-09-30"
VAL_START = "2025-10-01"
VAL_END = "2025-12-31"

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセット読み込みのエラーハンドリング
    try:
        train_ds = WindowedFeaturesAEDataset(
            samples_parquet=DATASET_SAMPLES,
            date_start=TRAIN_START,
            date_end=TRAIN_END,
        )
        val_ds = WindowedFeaturesAEDataset(
            samples_parquet=DATASET_SAMPLES,
            date_start=VAL_START,
            date_end=VAL_END,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Check if file exists: {DATASET_SAMPLES}")
        raise

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # infer feature_dim
    x0 = train_ds[0]
    feature_dim = int(x0.shape[1])
    print(f"Feature dim: {feature_dim}")

    # モデルハイパーパラメータ
    model_config = {
        "feature_dim": feature_dim,
        "seq_len": SEQ_LEN,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "latent_dim": LATENT_DIM,
    }

    model = TransformerAE(**model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=True
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_path = CKPT_DIR / "best.pt"
    patience_counter = 0
    patience_limit = 10

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_total = 0.0
        train_n = 0

        for xb in train_dl:
            xb = xb.to(device, non_blocking=True)  # (B,T,F)

            recon, _ = model(xb)
            loss = loss_fn(recon, xb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_total += float(loss.item()) * xb.size(0)
            train_n += xb.size(0)

        train_avg = train_total / max(train_n, 1)
        train_losses.append(train_avg)

        # Validation
        model.eval()
        val_total = 0.0
        val_n = 0

        with torch.no_grad():
            for xb in val_dl:
                xb = xb.to(device, non_blocking=True)
                recon, _ = model(xb)
                loss = loss_fn(recon, xb)
                val_total += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)

        val_avg = val_total / max(val_n, 1)
        val_losses.append(val_avg)

        # Display results
        improvement = "" if val_avg >= best_val_loss else " ★"
        print(f"epoch={epoch:03d} train_loss={train_avg:.6f} val_loss={val_avg:.6f} lr={opt.param_groups[0]['lr']:.6f}{improvement}")

        # Learning rate scheduling (based on validation loss)
        scheduler.step(val_avg)

        # Save best model ONLY based on validation loss (not training loss)
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_config": model_config,
                    "feature_dim": model_config["feature_dim"],
                    "seq_len": model_config["seq_len"],
                    "latent_dim": model_config["latent_dim"],
                    "train_loss": train_avg,
                    "val_loss": val_avg,
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
            print(f"  → Best validation model saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → No improvement (patience: {patience_counter}/{patience_limit})")

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch} (patience={patience_limit})")
            break

    # Save training history and metadata
    meta = {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "seq_len": SEQ_LEN,
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_trained": epoch,
        "best_val_loss": best_val_loss,
        "model_config": model_config,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint": str(best_path),
    }
    (CKPT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model selected based on VALIDATION performance (not training)")
    print(f"Saved: {best_path}")
    print(f"Meta : {CKPT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()