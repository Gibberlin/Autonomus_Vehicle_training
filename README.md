# CARLA Autonomous Vehicle Training and PilotNet Demos

This project trains and tests a camera-based PilotNet steering model in the CARLA
simulator. It includes a live driving demo, optional YOLO object detection and a
map-specific minimap, plus a fixed Town03 checkpoint route that combines
PilotNet steering with CARLA's navigation controller.

---

## 1. Overview

The project provides:

- **PilotNet steering** from the forward camera, with autopilot/model switching in
  the live demo.
- **Navigation-assisted driving** on a repeatable Town03 route: fixed spawn
  point 0, followed by checkpoints P1–P10. The hybrid mode weights navigation
  steering at 75% and PilotNet at 25%.
- **Optional YOLO detection**, lead-vehicle distance visualization, collision
  and lane-invasion counters, CSV telemetry, and video recording.
- **Optional map-specific minimap** with the active map's road network and a
  live ego-vehicle heading indicator, plus only two static markers: Start and
  Destination. The markers are selected from interior road samples.
- **Dataset capture and training** scripts for collecting steering data and
  training the PilotNet model.

## 2. Quick Start

### Requirements and setup

Use a CARLA server and Python API with matching versions. This repository
includes a Windows CARLA 0.9.16 wheel for CPython 3.12. From the repository root,
create an environment and install the project dependencies and matching wheel:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install .\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl
```

If you use a different Python version or operating system, install the CARLA
Python API wheel that matches your interpreter and CARLA server instead.

Start CARLA on its default RPC port (`2000`), then run commands from the
repository root. For the bundled Windows simulator:

```powershell
.\CarlaUE4.exe -carla-server -quality-level=Low -windowed -ResX=1280 -ResY=720
```

### Run a demo

The standard live demo starts in CARLA autopilot mode. Enable the minimap for the
selected map and run for two minutes:

```powershell
python live_demo.py --town Town03 --minimap --max-seconds 120
```

To test the fixed route with the KITTI controller and navigation from the start:

```powershell
python live_demo.py --start-model --minimap --vehicles 0 --no-yolo --max-seconds 120
```

Navigation is enabled by default and loads Town03 if needed, uses spawn point 0,
and plans the P1–P10 route from `config/checkpoints_town03.json`. Navigation
supplies 75% of model-mode steering and its route controller supplies speed and
hazard braking. Pass `--no-navigation` to disable it.

The dedicated route demo starts in hybrid PilotNet/navigation mode:

```powershell
python our_model_demo.py --vehicles 0 --no-yolo --max-seconds 120
```

In `live_demo.py`, press `m` to switch from autopilot to the KITTI controller
(hybrid when navigation is enabled), `a` to return to autopilot, and `q` or `Esc` to
quit. In `our_model_demo.py`, `m` selects hybrid steering, `r` selects the
navigation-only baseline, and `q` or `Esc` quits.

Both demos write telemetry under `results/`; `--video <path>` optionally saves
the camera output. The results directory is ignored by Git.

### Collect data and train

With CARLA available, capture a three-camera steering dataset and train the
model:

```powershell
python get_dataset.py --mode steer --samples 3000 --town Town03 --out drive_data
python train_steer.py --data drive_data --epochs 30
```

The legacy image-based `steer_model.pt` remains available for custom PilotNet
training, while the CARLA demos now default to the new KITTI feature-based
controller at `steer_model.pth`. Its normalisation parameters are loaded from
`steer_normalisation_params.csv` in the project root.

### Multimodal control model

For aligned RGB, depth, semantic segmentation, LiDAR, radar, vehicle telemetry,
traffic-light state, and speed-limit labels, run the synchronous recorder against
a CARLA server:

```powershell
python capture_fusion_dataset.py --towns Town01,Town03,Town05 --episodes-per-town 3 --out fusion_data
python train_fusion_model.py --data fusion_data --epochs 50
```

The recorder varies weather and traffic density and periodically adds a brief
steering disturbance, then records the Traffic Manager's recovery controls.
Samples are frame-synchronized and include timestamps. Training holds out whole
episodes for validation and writes `steer_fusion.pth`. Evaluate it in CARLA with
either demo using `--fusion-weights steer_fusion.pth`; this enables the matching
depth/semantic/LiDAR/radar sensors. The existing `steer_model.pth` remains the
default until a multimodal checkpoint has been collected and evaluated.

---

## 3. Project Directory Structure

```text
Autonomus_Vehicle_training/
│
├── config/
│   ├── checkpoints_town03.json    # 10 fixed checkpoints (P1-P10) coordinates, roads, GPS
│   └── *_roads_cache.json          # Map-specific vector road caches
│
├── navigation/
│   ├── __init__.py                 # Navigation module exports
│   ├── checkpoints.py             # Checkpoint dataclass, manager, loader, and CARLA generator
│   └── fixed_route.py              # Shared fixed Town03 route and steering blend
│
├── visualization/
│   ├── __init__.py                 # Visualization module exports
│   └── minimap.py                  # Minimap renderer, world-to-pixel projection, and OpenCV overlay
│
├── scripts/
│   ├── create_map_cache.py         # Utility to generate Town03 vector road cache
│   ├── generate_minimap_caches.py  # Generate caches and previews for CARLA's available maps
│   └── render_offline_minimap.py   # Pure Python standalone minimap preview generator
│
├── tests/
│   ├── test_minimap_offline.py       # 17 offline tests runnable without CARLA/GPU
│   ├── test_minimap_map_selection.py # Map cache selection and validation tests
│   └── test_model_navigation.py      # Route assembly and steering blend tests
│
├── generate_checkpoints.py         # Standalone CLI tool to generate/verify checkpoints
├── verify_carla_minimap.py         # CARLA integration & diagnostic test suite for the simulator machine
├── live_demo.py                    # Main simulation demo script with minimap integration
├── our_model_demo.py               # PilotNet + navigation on a fixed Town03 route
├── record_drive.py                 # Multi-camera dataset collection script
├── pilotnet.py                     # PilotNet steering CNN architecture & preprocessing
├── smoke_test.py                   # Minimal CARLA smoke test
├── train_steer.py                  # PyTorch steering model training script
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation and user guide
```

---

## 4. Dependencies

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

Run a CARLA server and install a matching Python API wheel on the demo machine.
The bundled Windows wheel is CARLA 0.9.16 for CPython 3.12. The simulator and
Python API versions must match.

On the machine running CARLA:
- CARLA server executable: `CarlaUE4.exe` (Windows) or `./CarlaUE4.sh` (Linux)
- Python API: Ensure the CARLA `.whl` or `PythonAPI/carla` is installed in the simulation environment.

---

## 5. The 10 Town03 Fixed Checkpoints

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

## 6. How to Generate or Regenerate Checkpoints

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

## 7. How to Run the Simulation with Minimap

### Basic Execution with Minimap

```bash
# Run live demo with the minimap for Town03
python live_demo.py --town Town03 --minimap
```

The minimap shows a Start marker snapped to an interior road sample near the
vehicle's spawn point and a well-separated Destination marker on the road
network. Markers are kept away from the minimap's outer bounds.

To load the tagged Guwahati OpenDRIVE map and generate its minimap preview:

```powershell
python LocalMap\load_guwahati.py
```

The script writes `results/minimap_previews/guwahati.png` and a map-specific
road cache. Its Start and Destination are selected from CARLA driving
waypoints, not from arbitrary image coordinates.

The demo selects the road cache using the active CARLA map name. If that map's
cache is missing, it builds and saves the cache from the live CARLA map. Caches
are stored in `config/` (`town03_roads_cache.json` for Town03 and
`<map>_roads_cache.json` for other maps). To pre-generate caches and 800px PNG
previews for every drivable map reported by CARLA:

```powershell
python scripts\generate_minimap_caches.py
```

The generator skips existing caches; pass `--force` to refresh them. When it
builds caches, it cycles the CARLA server through the required maps and restores
the map that was active when it started. PNG previews are written to
`results/minimap_previews/`; the `results/` directory is ignored by Git. To
generate only selected maps, pass their names with `--maps`, for example:

```powershell
python scripts\generate_minimap_caches.py --maps Town01 Town03 Town05_Opt
```

### Configurable Minimap Options

```bash
# Top-right corner (default), 260px panel, 85% opacity:
python live_demo.py --town Town03 --minimap --minimap-position top-right --minimap-size 260 --minimap-opacity 0.85

# Top-left corner:
python live_demo.py --town Town03 --minimap --minimap-position top-left --minimap-size 260 --minimap-opacity 0.85

# Use a custom Town03 route checkpoint file (navigation mode):
python live_demo.py --navigation --minimap --checkpoints-config config/checkpoints_town03.json

# Force refresh of road topology cache from live CARLA world:
python live_demo.py --town Town03 --minimap --refresh-map-cache
```

### Running Without Minimap (Existing Default Behavior)

```bash
# Normal simulation without minimap
python live_demo.py
```

### In-Simulation Controls
- `m`: Switch to KITTI controller steering (route-blended when navigation is enabled).
- `a`: Switch to CARLA Traffic Manager autopilot.
- `q` or `Esc`: Quit simulation and cleanly release actors and video writers.

---

## 8. Route-Guided KITTI Controller Test

`our_model_demo.py` runs a deterministic Town03 route from spawn point 0 through
checkpoints P1–P10 in `config/checkpoints_town03.json`. The default mode blends
KITTI controller steering (25%) with navigation steering (75%); press `r` for a pure
navigation-controller baseline or `m` to return to the blend.

`live_demo.py` enables navigation by default. With `--start-model`, it uses the
same fixed route and steering blend at startup.

```bash
# Plan and report the route without spawning vehicles or driving:
python our_model_demo.py --plan-only

# Run a short test without NPC traffic or YOLO:
python our_model_demo.py --vehicles 0 --no-yolo --max-seconds 120

# Use the same fixed route in the live demo, starting with PilotNet:
python live_demo.py --start-model --minimap --vehicles 0 --no-yolo --max-seconds 120
```

`our_model_demo.py` loads Town03 before planning. `live_demo.py --navigation`
loads Town03 only when it is not already active. Either can change the CARLA
server's current map.

---

## 9. Testing & Verification

### Local Offline Tests (No CARLA or GPU Required)

Run the comprehensive unit test suite on any development laptop:

```bash
python -m unittest discover -s tests
```

The 17 minimap tests verify:
- Checkpoint configuration integrity and uniqueness.
- World-to-pixel coordinate projection.
- Start and Destination selection from interior road points.
- Inverted vertical axis handling (North maps to top of image).
- Vehicle heading angle (yaw) to 2D pixel direction vectors.
- Aspect-ratio preservation (circular roundabouts remain circular).
- Road cache data validity and non-overlapping spatial distribution.
- Overlay positioning and adaptive resizing.

Four map-selection tests cover normalized names and paths, preferred-start
snapping, interior road-marker placement, and rejecting a cache for the wrong map.
Four model-navigation tests cover route segment chaining, empty route handling,
steering blending, and navigation-option parsing.

### Generate Standalone Offline Minimap Preview

```bash
python scripts/render_offline_minimap.py
```
Outputs `results/offline_minimap_preview.bmp`, showing the road layout, the two
road-snapped route markers, and the vehicle heading.

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

## 10. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `CARLA Python API unavailable` | CARLA egg/wheel not installed in environment | Install the CARLA `.whl` from your CARLA installation folder or append `PythonAPI/carla` to `PYTHONPATH`. |
| `Could not connect to CARLA server` | CarlaUE4 is not running or blocked by firewall | Start CARLA (`./CarlaUE4.sh` or `CarlaUE4.exe`) and check `--host` and `--port` (default 2000). |
| `Minimap overflows camera frame` | Minimap size exceeds small frame dimensions | The minimap automatically clamps to fit within the frame. You can also specify `--minimap-size 150` for smaller resolutions. |
| `OpenCV window not displaying` | Headless environment or missing display | Remove `--no-display` or ensure GUI environment is available. Video recording via `--video out.mp4` works in headless mode. |
| `Checkpoints out of sync` | Modified map version | Run `python generate_checkpoints.py --town Town03 --host localhost --port 2000` on the CARLA machine to refresh coordinates from the live map. |

