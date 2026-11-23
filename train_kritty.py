"""
Fine-tune the Kritty AVControlModel architecture on captured CARLA drive_data.

The capture contains speed, steering, throttle, and brake. Motion fields that were
not separately recorded are derived from consecutive center frames. Detection fields
are unavailable, so they remain zero for this dataset.
"""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "drive_data" / "labels.csv"
MODEL_DIR = ROOT
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FEATURES = [
    "speed_kmh",
    "forward_speed_mps",
    "heading_deg",
    "yaw_rate_rad_s",
    "forward_acceleration_mps2",
    "lateral_acceleration_mps2",
    "curvature_m_inv",
    "persons_detected",
    "bicycles_detected",
    "cars_detected",
    "trucks_detected",
    "buses_detected",
    "traffic_lights_detected",
    "total_detections",
]
OUTPUT_TARGETS = [
    "steering_angle_deg",
    "throttle_command",
    "brake_command",
]
HIDDEN_SIZES = [64, 32, 16]


class AVControlModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, target_scales=None):
        super().__init__()
        layers = []
        previous = input_size
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(0.05)])
            previous = hidden
        layers.append(nn.Linear(previous, output_size))
        self.network = nn.Sequential(*layers)
        if target_scales is None:
            target_scales = np.ones(output_size, dtype=np.float32)
        self.register_buffer(
            "target_scales", torch.tensor(target_scales, dtype=torch.float32)
        )

    def forward_norm(self, inputs):
        return self.network(inputs)

    def forward(self, inputs):
        return self.network(inputs) * self.target_scales


def build_dataset(frame):
    center = frame[frame["cam"].str.lower() == "center"].copy()
    center = center.sort_values("frame").reset_index(drop=True)

    speed_mps = center["speed"].astype(np.float32) / 3.6
    steering_control = center["steer"].astype(np.float32)
    # Preserve the KITTI sign convention in the training target:
    # positive = left, negative = right.
    steering_deg = -steering_control * 30.0

    frame_gap = center["frame"].diff().clip(lower=1)
    frame_seconds = frame_gap / 10.0
    speed_delta = speed_mps.diff().fillna(0.0)
    forward_accel = (speed_delta / frame_seconds).fillna(0.0).clip(-20.0, 20.0)

    steering_delta_deg = steering_deg.diff().fillna(0.0)
    yaw_rate_deg_s = steering_delta_deg / frame_seconds
    yaw_rate_deg_s = yaw_rate_deg_s.fillna(0.0).clip(-180.0, 180.0)
    yaw_rate_rad_s = np.deg2rad(yaw_rate_deg_s).astype(np.float32)
    heading_deg = np.rad2deg(np.cumsum(yaw_rate_rad_s) * 0.1).astype(np.float32)
    heading_deg = ((heading_deg + 180.0) % 360.0 - 180.0).astype(np.float32)

    curvature = (yaw_rate_rad_s / speed_mps.clip(lower=0.5)).clip(-1.0, 1.0)
    lateral_accel = yaw_rate_rad_s * speed_mps

    features = pd.DataFrame({
        "speed_kmh": center["speed"].astype(np.float32),
        "forward_speed_mps": speed_mps,
        "heading_deg": heading_deg,
        "yaw_rate_rad_s": yaw_rate_rad_s,
        "forward_acceleration_mps2": forward_accel,
        "lateral_acceleration_mps2": lateral_accel,
        "curvature_m_inv": curvature,
        "persons_detected": np.zeros(len(center), dtype=np.float32),
        "bicycles_detected": np.zeros(len(center), dtype=np.float32),
        "cars_detected": np.zeros(len(center), dtype=np.float32),
        "trucks_detected": np.zeros(len(center), dtype=np.float32),
        "buses_detected": np.zeros(len(center), dtype=np.float32),
        "traffic_lights_detected": np.zeros(len(center), dtype=np.float32),
        "total_detections": np.zeros(len(center), dtype=np.float32),
    })
    targets = pd.DataFrame({
        "steering_angle_deg": steering_deg,
        "throttle_command": center["throttle"].clip(0.0, 1.0).astype(np.float32),
        "brake_command": center["brake"].clip(0.0, 1.0).astype(np.float32),
    })
    return features, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"CARLA capture not found: {data_path}")

    frame = pd.read_csv(data_path)
    features, targets = build_dataset(frame)
    count = len(features)
    cut = count - int(count * args.val_frac)

    mean = features.iloc[:cut].mean(axis=0).to_numpy(dtype=np.float32)
    std = features.iloc[:cut].std(axis=0).to_numpy(dtype=np.float32)
    std[std < 1e-6] = 1.0
    normalised = (features.to_numpy(dtype=np.float32) - mean) / std

    target_scales = targets.iloc[:cut].std(axis=0).to_numpy(
        dtype=np.float32, copy=True
    )
    target_scales[target_scales < 1e-6] = 1.0
    target_normalised = targets.iloc[:cut].to_numpy(dtype=np.float32) / target_scales

    x_train = torch.tensor(normalised[:cut], dtype=torch.float32)
    y_train = torch.tensor(
        targets.iloc[:cut].to_numpy(dtype=np.float32) / target_scales,
        dtype=torch.float32,
    )
    x_val = torch.tensor(normalised[cut:], dtype=torch.float32)
    y_val = torch.tensor(
        targets.iloc[cut:].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = AVControlModel(
        len(INPUT_FEATURES),
        HIDDEN_SIZES,
        len(OUTPUT_TARGETS),
        target_scales=target_scales,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_loss = float("inf")
    best_path = MODEL_DIR / "steer_model.pth"
    backup_path = MODEL_DIR / "steer_model.pre_kritty.pth"
    if best_path.is_file():
        shutil.copy2(best_path, backup_path)

    print(f"Training on {cut} frames, validating on {count - cut} frames")
    print(f"Device: {device}")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model.forward_norm(x_batch)
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        train_loss /= cut

        model.eval()
        with torch.no_grad():
            val_normalised = model(x_val.to(device)).cpu().numpy()
        val_loss = np.mean(
            (val_normalised - y_val.numpy()) ** 2
        )
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(
                f"epoch {epoch + 1:3d}/{args.epochs} "
                f"train={train_loss:.6f} val={val_loss:.6f} *saved"
            )
        elif (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1:3d}/{args.epochs} "
                f"train={train_loss:.6f} val={val_loss:.6f}"
            )

    pd.DataFrame({
        "feature": INPUT_FEATURES,
        "mean": mean,
        "std": std,
    }).to_csv(MODEL_DIR / "steer_normalisation_params.csv", index=False)
    pd.DataFrame({
        "target": OUTPUT_TARGETS,
        "scale": target_scales,
    }).to_csv(MODEL_DIR / "steer_target_scales.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    scripted = torch.jit.trace(model.cpu(), torch.zeros(1, len(INPUT_FEATURES)))
    scripted.save(str(MODEL_DIR / "steer_model_ts.pt"))

    try:
        torch.onnx.export(
            model.cpu(),
            torch.zeros(1, len(INPUT_FEATURES)),
            str(MODEL_DIR / "steer_model.onnx"),
            input_names=["sensor_inputs"],
            output_names=["control_outputs"],
            dynamic_axes={
                "sensor_inputs": {0: "batch_size"},
                "control_outputs": {0: "batch_size"},
            },
            opset_version=18,
            dynamo=False,
        )
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")

    print(f"Saved fine-tuned weights: {best_path}")
    print(f"Saved TorchScript model: {MODEL_DIR / 'steer_model_ts.pt'}")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Previous KITTI checkpoint: {backup_path}")


if __name__ == "__main__":
    main()
