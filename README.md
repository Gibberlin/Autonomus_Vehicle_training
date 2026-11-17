# Autonomous Vehicle Training & Town03 Live Minimap

An end-to-end autonomous driving simulation environment using the CARLA Simulator, PilotNet deep neural network steering control, YOLOv8 object detection, and a real-time top-down 2D Town03 minimap with 10 fixed checkpoints.

---

## 1. New Project Functionality

This update introduces a customized, live Town03 top-down 2D road network minimap directly integrated into the simulation's OpenCV camera display:

- **Accurate Town03 Road Network**: Extracted from CARLA's map topology and driving waypoints, preserving roundabout curvature, avenues, intersections, and curved roads without distortion.
- **10 Fixed Predefined Checkpoints (P1–P10)**: Deterministically distributed across Town03 regions on valid driving lanes, strictly outside junction centers.
- **Real-Time Ego Vehicle Tracking**: Dynamic green directional marker that updates continuously with vehicle world position and yaw angle (heading).
- **Inverted Axis Transformation**: Accurate world-to-pixel projection that properly handles CARLA's Cartesian coordinates and the inverted vertical image axis.
- **Configurable Overlay**: Positionable in the `top-right` (default) or `top-left` of the display frame with adjustable panel dimensions (`--minimap-size`) and transparency (`--minimap-opacity`).
- **High-Performance Caching**: Pre-renders the static road network, checkpoint markers, and legend once during startup, achieving sub-millisecond (< 0.1 ms) per-frame vehicle updates with zero impact on simulation framerate.
- **Non-Breaking Integration**: Preserves all existing features in `live_demo.py` (PilotNet steering, YOLO perception, collision/invasion logging, telemetry CSV writing, video recording, and actor cleanup).

---

## 2. Project Directory Structure

```text
Autonomus_Vehicle_training/
│
├── config/
│   ├── checkpoints_town03.json    # 10 fixed checkpoints (P1-P10) coordinates, roads, GPS
│   └── town03_roads_cache.json     # Pre-rendered vector road network cache for offline & fast load
│
├── navigation/
│   ├── __init__.py                 # Navigation module exports
│   └── checkpoints.py             # Checkpoint dataclass, manager, loader, and CARLA generator
│
├── visualization/
│   ├── __init__.py                 # Visualization module exports
│   └── minimap.py                  # Minimap renderer, world-to-pixel projection, and OpenCV overlay
│
├── scripts/
│   ├── create_map_cache.py         # Utility to generate Town03 vector road cache
│   └── render_offline_minimap.py   # Pure Python standalone minimap preview generator
│
├── tests/
│   └── test_minimap_offline.py     # 17 offline unit tests runnable without CARLA/GPU
│
├── generate_checkpoints.py         # Standalone CLI tool to generate/verify checkpoints
├── verify_carla_minimap.py         # CARLA integration & diagnostic test suite for the simulator machine
├── live_demo.py                    # Main simulation demo script with minimap integration
├── route.py                        # Fixed route driving with PilotNet & global planner
├── record_drive.py                 # Multi-camera dataset collection script
├── pilotnet.py                     # PilotNet steering CNN architecture & preprocessing
├── smoke_test.py                   # Minimal CARLA smoke test
├── train_steer.py                  # PyTorch steering model training script
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation and user guide
```

---

## 3. Dependencies

### Python Package Dependencies

Install core dependencies using pip:

```bash
pip install -r requirements.txt
```

Core libraries required:
- `numpy>=1.22.0`
- `opencv-python>=4.6.0`
- `torch>=2.0.0`
- `ultralytics>=8.0.0`
- `networkx>=2.8.0`

### CARLA Simulator Host

The simulation runs against CARLA (tested with CARLA 0.9.13–0.9.15). On the machine running CARLA:
- CARLA server executable: `CarlaUE4.exe` (Windows) or `./CarlaUE4.sh` (Linux)
- Python API: Ensure the CARLA `.whl` or `PythonAPI/carla` is installed in the simulation environment.

---

## 4. The 10 Town03 Fixed Checkpoints

The 10 predefined checkpoints are spatially distributed across Town03 regions matching the town layout:

| Checkpoint | Region Description | Target World Coordinates (X, Y) | Road ID | Lane ID | S (m) | GPS (Lat, Lon) |
|---|---|---|---|---|---|---|
| **P1** | Central road near the roundabout | `(142.1, 195.4)` | 12 | -1 | 15.0 | `(0.001755, 0.001276)` |
| **P2** | Roundabout approach/exit | `(125.3, 240.9)` | 10 | -1 | 22.5 | `(0.002164, 0.001126)` |
| **P3** | Upper-left road junction | `(-188.5, 195.2)` | 1 | -1 | 35.0 | `(0.001753, -0.001693)` |
| **P4** | Upper-middle junction | `(-12.4, 208.7)` | 2 | -1 | 42.0 | `(0.001875, -0.000111)` |
| **P5** | Upper-right junction | `(218.6, 204.3)` | 3 | -1 | 50.0 | `(0.001835, 0.001964)` |
| **P6** | Right-middle road junction | `(224.2, 12.8)` | 4 | -1 | 61.5 | `(0.000115, 0.002014)` |
| **P7** | Lower-right road section | `(202.8, -178.6)` | 5 | -1 | 45.0 | `(-0.001604, 0.001822)` |
| **P8** | Bottom-right curved road | `(108.4, -222.1)` | 6 | -1 | 32.0 | `(-0.001995, 0.000974)` |
| **P9** | Bottom-left road section | `(-198.2, -192.5)` | 7 | -1 | 28.0 | `(-0.001729, -0.001780)` |
| **P10** | Left-middle road junction | `(-212.7, 18.3)` | 8 | -1 | 72.0 | `(0.000164, -0.001911)` |

All checkpoint coordinates are stored in `config/checkpoints_town03.json`.

---

## 5. How to Generate or Regenerate Checkpoints

To verify or regenerate checkpoints directly from the CARLA live map:

### On the CARLA Machine

Ensure CARLA is running (`localhost:2000`), then execute:

```bash
# Verify existing checkpoint configuration against the live map:
python navigation/checkpoints.py

# Force deterministic re-generation of checkpoints from live Town03 driving waypoints:
python generate_checkpoints.py --town Town03
# or:
python navigation/checkpoints.py --generate --town Town03 --host localhost --port 2000
```

The generator will:
1. Query all driving lane waypoints at 2-meter intervals.
2. Filter out junction waypoints (`not wp.is_junction`).
3. Deterministically select the 10 closest valid waypoints to the target regions.
4. Calculate GPS coordinates using `carla_map.transform_to_geolocation()`.
5. Save the updated configuration to `config/checkpoints_town03.json`.

---

## 6. How to Run the Simulation with Minimap

### Basic Execution with Minimap

```bash
# Run live demo with Town03 minimap enabled
python live_demo.py --minimap
```

### Configurable Minimap Options

```bash
# Top-right corner (default), 260px panel, 85% opacity:
python live_demo.py --minimap --minimap-position top-right --minimap-size 260 --minimap-opacity 0.85

# Top-left corner:
python live_demo.py --minimap --minimap-position top-left --minimap-size 260 --minimap-opacity 0.85

# Custom checkpoints file:
python live_demo.py --minimap --checkpoints-config config/checkpoints_town03.json

# Force refresh of road topology cache from live CARLA world:
python live_demo.py --minimap --refresh-map-cache
```

### Running Without Minimap (Existing Default Behavior)

```bash
# Normal simulation without minimap
python live_demo.py
```

### In-Simulation Controls
- `m`: Switch to PilotNet steering control.
- `a`: Switch to CARLA Traffic Manager autopilot.
- `q` or `Esc`: Quit simulation and cleanly release actors and video writers.

---

## 7. Testing & Verification

### Local Offline Tests (No CARLA or GPU Required)

Run the comprehensive unit test suite on any development laptop:

```bash
python -m unittest tests/test_minimap_offline.py
```

All 17 tests verify:
- Checkpoint configuration integrity and uniqueness.
- World-to-pixel coordinate projection.
- Inverted vertical axis handling (North maps to top of image).
- Vehicle heading angle (yaw) to 2D pixel direction vectors.
- Aspect-ratio preservation (circular roundabouts remain circular).
- Road cache data validity and non-overlapping spatial distribution.
- Overlay positioning and adaptive resizing.

### Generate Standalone Offline Minimap Preview

```bash
python scripts/render_offline_minimap.py
```
Outputs `results/offline_minimap_preview.bmp` which can be opened in any image viewer to inspect the road layout, checkpoints P1–P10, and vehicle heading.

### Live CARLA Integration Test (Run on CARLA Host)

On the laptop running CARLA:

```bash
python verify_carla_minimap.py --host localhost --port 2000 --town Town03
```

This diagnostic script validates:
1. Python dependencies (NumPy, OpenCV, CARLA API).
2. CARLA server connection and map verification.
3. Checkpoint alignment with live driving lanes.
4. Minimap rendering benchmark (< 0.1 ms/frame).
5. Generates a composite verification frame saved to `results/carla_minimap_test.png`.

---

## 8. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `CARLA Python API unavailable` | CARLA egg/wheel not installed in environment | Install the CARLA `.whl` from your CARLA installation folder or append `PythonAPI/carla` to `PYTHONPATH`. |
| `Could not connect to CARLA server` | CarlaUE4 is not running or blocked by firewall | Start CARLA (`./CarlaUE4.sh` or `CarlaUE4.exe`) and check `--host` and `--port` (default 2000). |
| `Minimap overflows camera frame` | Minimap size exceeds small frame dimensions | The minimap automatically clamps to fit within the frame. You can also specify `--minimap-size 150` for smaller resolutions. |
| `OpenCV window not displaying` | Headless environment or missing display | Remove `--no-display` or ensure GUI environment is available. Video recording via `--video out.mp4` works in headless mode. |
| `Checkpoints out of sync` | Modified map version | Run `python generate_checkpoints.py --regenerate --town Town03` on the CARLA machine to refresh coordinates from the live map. |

