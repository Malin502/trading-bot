from __future__ import annotations
import sys

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tempfile
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.intraday_1h_ae.datasets_ae_1h import WindowedFeaturesAEDataset
from model.TransformerAE import TransformerAE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_SAMPLES = PROJECT_ROOT / "datasets/intraday_1h_ae/samples.parquet"
CKPT_PATH = PROJECT_ROOT / "model/checkpoints/ae_1h/best.pt"
OUT_PATH = PROJECT_ROOT / "datasets/intraday_1h_ae/market_latent32.parquet"

SEQ_LEN = 56
LATENT_DIM = 32
BATCH_SIZE = 512
NUM_WORKERS = 2

MARKET_TICKER = "1306.T"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1) samples.parquet から 1306.T だけ抜く
    manifest_all = pd.read_parquet(DATASET_SAMPLES).reset_index(drop=True)
    if "ticker" not in manifest_all.columns:
        raise KeyError(f"'ticker' column not found in {DATASET_SAMPLES}")

    manifest_mkt = manifest_all[manifest_all["ticker"] == MARKET_TICKER].reset_index(drop=True)
    if len(manifest_mkt) == 0:
        raise ValueError(f"No rows for ticker={MARKET_TICKER} in {DATASET_SAMPLES}")

    # Dataset が parquet path を要求するので、一時ファイルに書いて渡す（確実に動く方式）
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "market_samples_tmp.parquet"
        manifest_mkt.to_parquet(tmp_path, index=False)

        ds = WindowedFeaturesAEDataset(samples_parquet=tmp_path)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # ---- 2) モデルロード
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        state = ckpt.get("model_state")
        if state is None:
            raise KeyError(f"Checkpoint missing 'model_state'. keys={list(ckpt.keys())}")

        x0 = ds[0]  # (T, F)
        feature_dim = int(x0.shape[1])

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

        # ---- 3) encode
        latents = []
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(device, non_blocking=True)
                z = model.encode(xb)  # (B, 32)
                latents.append(z.detach().cpu().numpy())

        Z = np.concatenate(latents, axis=0)

    # ---- 4) mkt_z** を付与して保存
    if len(manifest_mkt) != Z.shape[0]:
        raise ValueError(f"Row count mismatch: manifest={len(manifest_mkt)} vs latents={Z.shape[0]}")

    out = manifest_mkt.copy()
    for j in range(LATENT_DIM):
        out[f"mkt_z{j:02d}"] = Z[:, j].astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} rows={len(out)} cols={out.shape[1]}")


if __name__ == "__main__":
    main()
