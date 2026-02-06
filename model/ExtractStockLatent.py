from __future__ import annotations
import sys

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.intraday_1h_ae.datasets_ae_1h import WindowedFeaturesAEDataset
from model.TransformerAE import TransformerAE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_SAMPLES = PROJECT_ROOT / "datasets/intraday_1h_ae/samples.parquet"
CKPT_PATH = PROJECT_ROOT / "model/checkpoints/ae_1h/best.pt"
OUT_PATH = PROJECT_ROOT / "datasets/intraday_1h_ae/stock_latent32.parquet"

SEQ_LEN = 56
LATENT_DIM = 32
BATCH_SIZE = 512
NUM_WORKERS = 2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load full samples (no date filter)
    ds = WindowedFeaturesAEDataset(samples_parquet=DATASET_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    x0 = ds[0]  # (T, F)
    feature_dim = int(x0.shape[1])

    state = ckpt.get("model_state")  # あなたの保存形式に合わせる
    if state is None:
        raise KeyError(f"Checkpoint missing 'model_state'. keys={list(ckpt.keys())}")

    model = TransformerAE(
        feature_dim=feature_dim,
        seq_len=SEQ_LEN,
        latent_dim=LATENT_DIM,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    latents = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            z = model.encode(xb)  # (B,32)
            latents.append(z.detach().cpu().numpy())

    Z = np.concatenate(latents, axis=0)  # (num_samples, 32)

    # attach to original manifest rows
    manifest = pd.read_parquet(DATASET_SAMPLES).reset_index(drop=True)
    for j in range(LATENT_DIM):
        manifest[f"z{j:02d}"] = Z[:, j].astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(OUT_PATH)
    print(f"Saved: {OUT_PATH} rows={len(manifest)} cols={manifest.shape[1]}")


if __name__ == "__main__":
    main()
