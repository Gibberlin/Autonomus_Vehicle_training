import json
from queue import Queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from capture_fusion_dataset import decode_depth, decode_semantic, receive_frame
from fused_control_model import (
    FusedControlDataset,
    FusedControlModel,
    STATE_FEATURES,
)


class TestFusedControlModel(unittest.TestCase):
    def test_model_fuses_all_modalities_into_three_controls(self):
        model = FusedControlModel().eval()
        with torch.no_grad():
            controls = model(
                torch.zeros(2, 5, 90, 160),
                torch.zeros(2, 2048, 4),
                torch.zeros(2, 128, 4),
                torch.zeros(2, len(STATE_FEATURES)),
            )
        self.assertEqual(tuple(controls.shape), (2, 3))
        self.assertTrue(torch.isfinite(controls).all())

    def test_dataset_reads_aligned_modalities_and_control_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("rgb", "depth", "semantic", "lidar", "radar"):
                (root / name).mkdir()
            cv2.imwrite(
                str(root / "rgb" / "frame.jpg"),
                np.full((90, 160, 3), 127, dtype=np.uint8),
            )
            np.save(root / "depth" / "frame.npy", np.full((90, 160), 12, np.float32))
            np.save(root / "semantic" / "frame.npy", np.full((90, 160), 7, np.uint8))
            np.save(root / "lidar" / "frame.npy", np.array([[10, 0, 0, 0.5]], np.float32))
            np.save(root / "radar" / "frame.npy", np.array([[20, 0, 0, -5]], np.float32))
            row = {
                "rgb": "rgb/frame.jpg",
                "depth": "depth/frame.npy",
                "semantic": "semantic/frame.npy",
                "lidar": "lidar/frame.npy",
                "radar": "radar/frame.npy",
                "steer": 0.2,
                "throttle": 0.4,
                "brake": 0.0,
            }
            row.update({name: 0.0 for name in STATE_FEATURES})
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            dataset = FusedControlDataset(
                manifest,
                np.zeros(len(STATE_FEATURES), np.float32),
                np.ones(len(STATE_FEATURES), np.float32),
            )
            images, lidar, radar, state, target = dataset[0]

        self.assertEqual(tuple(images.shape), (5, 90, 160))
        self.assertAlmostEqual(float(lidar[0, 0]), 0.2)
        self.assertAlmostEqual(float(radar[0, 0]), 0.25)
        self.assertEqual(tuple(state.shape), (len(STATE_FEATURES),))
        self.assertTrue(np.allclose(target.numpy(), [0.2, 0.4, 0.0]))

    def test_sensor_decoders_and_frame_matching(self):
        bgra = np.zeros((1, 2, 4), dtype=np.uint8)
        bgra[0, 0, 2] = 3
        bgra[0, 1, 1] = 1
        bgra[0, 1, 2] = 2
        depth = decode_depth(bgra.tobytes(), 1, 2)
        semantic = decode_semantic(bgra.tobytes(), 1, 2)
        self.assertAlmostEqual(float(depth[0, 0]), 3.0 * 1000 / 16777215, places=8)
        self.assertAlmostEqual(float(depth[0, 1]), 258.0 * 1000 / 16777215, places=6)
        self.assertTrue(np.array_equal(semantic, [[3, 2]]))

        sensor_queue = Queue()
        sensor_queue.put(SimpleNamespace(frame=4))
        sensor_queue.put(SimpleNamespace(frame=5))
        self.assertEqual(receive_frame(sensor_queue, 5, 0.1).frame, 5)


if __name__ == "__main__":
    unittest.main()
