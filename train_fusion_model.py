"""Train the fused camera, LiDAR, radar, and vehicle-state control model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fused_control_model import (
    FusedControlDataset,
    FusedControlModel,
    STATE_FEATURES,
)


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "fusion_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "steer_fusion.pth")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = args.data / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    rows = pd.read_json(manifest, lines=True)
    if len(rows) < 10:
        raise ValueError("At least 10 synchronized samples are required")
    required = set(STATE_FEATURES) | {
        "episode_id", "rgb", "depth", "semantic", "lidar", "radar",
        "steer", "throttle", "brake",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")

    # Hold out complete episodes when possible, preventing adjacent-frame leakage.
    episode_ids = sorted(rows["episode_id"].unique())
    if len(episode_ids) > 1:
        val_ids = set(episode_ids[-max(1, round(len(episode_ids) * 0.2)):])
        train_indices = rows.index[~rows["episode_id"].isin(val_ids)].tolist()
        val_indices = rows.index[rows["episode_id"].isin(val_ids)].tolist()
    else:
        split = int(len(rows) * 0.8)
        train_indices = list(range(split))
        val_indices = list(range(split, len(rows)))
    if not train_indices or not val_indices:
        raise ValueError("Dataset split produced an empty train or validation set")

    training_rows = rows.loc[train_indices]
    state_mean = training_rows[list(STATE_FEATURES)].mean().to_numpy(np.float32)
    state_std = training_rows[list(STATE_FEATURES)].std().to_numpy(np.float32)
    state_std[state_std < 1e-6] = 1.0
    target_columns = ["steer", "throttle", "brake"]
    target_scales = training_rows[target_columns].std().to_numpy(
        dtype=np.float32, copy=True
    )
    target_scales[target_scales < 1e-6] = 1.0

    dataset = FusedControlDataset(
        manifest, state_mean, state_std, seed=args.seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusedControlModel().to(device)
    model.target_scales.copy_(torch.tensor(target_scales, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    criterion = nn.SmoothL1Loss()
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Samples: {len(train_indices)} train / {len(val_indices)} validation; "
        f"{len(episode_ids)} episodes; device={device}"
    )
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, lidar, radar, state, target in train_loader:
            images, lidar, radar = images.to(device), lidar.to(device), radar.to(device)
            state, target = state.to(device), target.to(device)
            prediction = model(images, lidar, radar, state)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(images)
        train_loss /= len(train_indices)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, lidar, radar, state, target in val_loader:
                prediction = model(
                    images.to(device), lidar.to(device), radar.to(device),
                    state.to(device),
                )
                validation_loss += criterion(
                    prediction, target.to(device)
                ).item() * len(images)
        validation_loss /= len(val_indices)

        if validation_loss < best_loss:
            best_loss = validation_loss
            checkpoint_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save({
                "model_state": checkpoint_state,
                "state_mean": torch.tensor(state_mean),
                "state_std": torch.tensor(state_std),
                "target_scales": torch.tensor(target_scales),
                "state_features": list(STATE_FEATURES),
                "episodes": episode_ids,
                "format_version": 1,
            }, args.output)
            model.to(device)
            marker = " *saved"
        else:
            marker = ""
        print(
            f"epoch {epoch + 1:3d}/{args.epochs} "
            f"train={train_loss:.5f} val={validation_loss:.5f}{marker}"
        )

    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Saved multimodal checkpoint: {args.output}")
    print(
        "This is a supervised CARLA simulator model; validate extensively in "
        "simulation before any deployment."
    )


if __name__ == "__main__":
    main()
