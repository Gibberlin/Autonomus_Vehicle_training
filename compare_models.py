"""Compare the previous tabular steering model with the trained fusion model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from fused_control_model import (
    FusedControlDataset,
    FusedControlModel,
    STATE_FEATURES,
    episode_split_indices,
)
from kritty_motion_controller import (
    INPUT_FEATURES,
    KrittySteeringController,
)


ROOT = Path(__file__).resolve().parent
TARGETS = ("steer", "throttle", "brake")


def previous_model_features(row):
    features = {
        "speed_kmh": row["speed_kmh"],
        "forward_speed_mps": row["forward_speed_mps"],
        "heading_deg": row["heading_deg"],
        "yaw_rate_rad_s": row["yaw_rate_rad_s"],
        "forward_acceleration_mps2": row["forward_acceleration_mps2"],
        "lateral_acceleration_mps2": row["lateral_acceleration_mps2"],
        "curvature_m_inv": (
            row["yaw_rate_rad_s"] / row["forward_speed_mps"]
            if abs(row["forward_speed_mps"]) > 0.5 else 0.0
        ),
        "persons_detected": 0.0,
        "bicycles_detected": 0.0,
        "cars_detected": 0.0,
        "trucks_detected": 0.0,
        "buses_detected": 0.0,
        "traffic_lights_detected": 0.0,
        "total_detections": 0.0,
    }
    return np.array([features[name] for name in INPUT_FEATURES], dtype=np.float32)


def control_outputs(model, normalized_features, device):
    tensor = torch.from_numpy(normalized_features).to(device)
    with torch.no_grad():
        physical = model.model(tensor).cpu().numpy()
    steering = np.clip(-physical[:, 0] / 30.0, -1.0, 1.0)
    throttle = np.clip(physical[:, 1], 0.0, 1.0)
    brake = np.clip(physical[:, 2], 0.0, 1.0)
    throttle[brake > 0.15] = 0.0
    return np.stack((steering, throttle, brake), axis=1)


def summarize(truth, predictions):
    result = {}
    for model_name, values in predictions.items():
        errors = values - truth
        entry = {}
        for index, target in enumerate(TARGETS):
            entry[target] = {
                "mae": float(np.mean(np.abs(errors[:, index]))),
                "rmse": float(np.sqrt(np.mean(errors[:, index] ** 2))),
                "bias": float(np.mean(errors[:, index])),
            }
        turning = np.abs(truth[:, 0]) >= 0.05
        entry["steering_direction_accuracy"] = (
            float(np.mean(np.sign(values[turning, 0]) == np.sign(truth[turning, 0])))
            if np.any(turning) else None
        )
        result[model_name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "fusion_data")
    parser.add_argument("--previous-weights", type=Path, default=ROOT / "steer_model.pth")
    parser.add_argument(
        "--previous-normalisation",
        type=Path,
        default=ROOT / "steer_normalisation_params.csv",
    )
    parser.add_argument("--fusion-weights", type=Path, default=ROOT / "steer_fusion.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "fusion_comparison")
    args = parser.parse_args()

    manifest = args.data / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    for path in (args.previous_weights, args.previous_normalisation, args.fusion_weights):
        if not path.is_file():
            raise FileNotFoundError(f"Model comparison input not found: {path}")

    rows = pd.read_json(manifest, lines=True)
    missing = set(STATE_FEATURES + TARGETS + ("episode_id",)) - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    _, validation_indices = episode_split_indices(rows["episode_id"])
    if not validation_indices:
        raise ValueError("No validation samples are available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous = KrittySteeringController(
        args.previous_weights, args.previous_normalisation, device
    )
    previous_features = np.stack([
        previous_model_features(rows.iloc[index])
        for index in validation_indices
    ])
    previous_features = (previous_features - previous.mean) / previous.std
    previous_predictions = np.concatenate([
        control_outputs(previous, batch, device)
        for batch in np.array_split(
            previous_features,
            max(1, int(np.ceil(len(previous_features) / 256))),
        )
    ])

    checkpoint = torch.load(args.fusion_weights, map_location=device, weights_only=True)
    if checkpoint.get("state_features") != list(STATE_FEATURES):
        raise ValueError(f"Unsupported state feature schema in {args.fusion_weights}")
    current = FusedControlModel().to(device)
    current.load_state_dict(checkpoint["model_state"])
    current.eval()
    dataset = FusedControlDataset(
        manifest,
        checkpoint["state_mean"].cpu().numpy(),
        checkpoint["state_std"].cpu().numpy(),
    )
    loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )
    fused_outputs = []
    with torch.no_grad():
        for images, lidar, radar, state, _targets in loader:
            output = current(
                images.to(device), lidar.to(device), radar.to(device), state.to(device)
            )
            output = output.cpu().numpy()
            output[:, 1] = np.clip(output[:, 1], 0.0, 1.0)
            output[:, 2] = np.clip(output[:, 2], 0.0, 1.0)
            output[output[:, 2] > 0.15, 1] = 0.0
            output[:, 0] = np.clip(output[:, 0], -1.0, 1.0)
            fused_outputs.append(output)
    fused_predictions = np.concatenate(fused_outputs)
    truth = rows.iloc[validation_indices][list(TARGETS)].to_numpy(np.float32)
    predictions = {
        "previous_tabular": previous_predictions,
        "new_multimodal": fused_predictions,
    }
    metrics = summarize(truth, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    validation_rows = rows.iloc[validation_indices].reset_index(drop=True)
    for index, row in validation_rows.iterrows():
        output_rows.append({
            "episode_id": row["episode_id"],
            "frame": row["frame"],
            "truth_steer": truth[index, 0],
            "truth_throttle": truth[index, 1],
            "truth_brake": truth[index, 2],
            "previous_steer": previous_predictions[index, 0],
            "previous_throttle": previous_predictions[index, 1],
            "previous_brake": previous_predictions[index, 2],
            "new_steer": fused_predictions[index, 0],
            "new_throttle": fused_predictions[index, 1],
            "new_brake": fused_predictions[index, 2],
        })
    pd.DataFrame(output_rows).to_csv(args.output.with_suffix(".csv"), index=False)
    report = {
        "validation_samples": len(validation_indices),
        "validation_episodes": sorted(rows.iloc[validation_indices]["episode_id"].unique()),
        "metrics": metrics,
        "prediction_csv": str(args.output.with_suffix(".csv")),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
