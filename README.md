# CARLA Autonomous Vehicle Training

This repository contains CARLA simulation demos, steering controllers, dataset
capture tools, and map/navigation utilities. The models and control examples are
for simulator research and testing only; they are not validated for real-road or
unsupervised vehicle use.

## What is in the project

- **Live CARLA demo** ([live_demo.py](./live_demo.py)): switches between CARLA
  Traffic Manager autopilot and the project steering controller; supports
  navigation, YOLO detections, telemetry, video, and the minimap.
- **Fixed-route demo** ([our_model_demo.py](./our_model_demo.py)): runs the
  Town03 route through checkpoints P1–P10, blending model and navigation
  steering.
- **Current tabular controller** ([kritty_motion_controller.py](./kritty_motion_controller.py)):
  loads `steer_model.pth` and its `steer_normalisation_params.csv` statistics.
- **Optional multimodal controller** ([fused_control_model.py](./fused_control_model.py)):
  combines camera images, LiDAR, radar, and vehicle/rule-state inputs. Collect
  data and train a checkpoint before enabling it.
- **Legacy PilotNet** ([pilotnet.py](./pilotnet.py)): camera-only model and
  preprocessing used by the legacy steering training workflow.
- **Navigation and minimap** ([navigation/](./navigation),
  [visualization/](./visualization)): fixed Town03 route planning, checkpoint
  configuration, and map-specific road overlays.

## Setup

Use a CARLA server and Python API version that match. The bundled Windows CARLA
0.9.16 wheel is for CPython 3.12. From the repository root, for example:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pandas matplotlib
python -m pip install .\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl
```

Install the matching CARLA wheel for your Python version and operating system if
you do not use the bundled Windows wheel. `pandas` and `matplotlib` are used by
the training and reporting scripts.

Start the simulator, then run Python commands from this repository:

```powershell
.\CarlaUE4.exe -carla-server -quality-level=Low -windowed -ResX=1280 -ResY=720
```

The standard RPC port is `2000`; Traffic Manager defaults to port `8000`.

## Run the demos

### Live demo

Navigation is enabled by default and uses the fixed Town03 route. The demo starts
in CARLA autopilot mode; press `m` to hand control to the project model, `a` to
return to autopilot, and `q` or `Esc` to quit.

```powershell
# Start in autopilot; navigate Town03 and show the minimap
python live_demo.py --town Town03 --minimap

# Start with the project model, no NPCs or YOLO, for a short run
python live_demo.py --start-model --vehicles 0 --no-yolo --max-seconds 120

# Disable the fixed-route navigation behavior
python live_demo.py --no-navigation
```

`--start-model` starts with the model rather than autopilot. `--navigation` and
`--no-navigation` explicitly enable or disable navigation. Navigation currently
supports Town03 and uses spawn point 0 plus the checkpoints in
`config/checkpoints_town03.json`. Navigation contributes 75% of hybrid steering;
the model contributes 25%. Navigation provides route speed and hazard braking.

Useful live-demo options include:

- `--host`, `--port`, `--tm-port`: CARLA and Traffic Manager connection settings.
- `--vehicles`: number of NPC vehicles.
- `--target-speed`: model-mode target speed in km/h.
- `--no-yolo`: disable object detection.
- `--video <path>`: save camera video.
- `--no-display`: run without an OpenCV window.
- `--minimap`, `--minimap-position`, `--minimap-size`, `--minimap-opacity`:
  show and configure the road minimap.
- `--checkpoints-config <path>`: use a custom Town03 route configuration.
- `--refresh-map-cache`: rebuild the active map's road cache.

Telemetry and demo results are written under `results/`.

### Fixed Town03 route demo

This demo starts in model-plus-navigation mode. Press `r` for navigation-only,
`m` to return to the blend, and `q` or `Esc` to quit.

```powershell
# Plan the route without spawning vehicles or driving
python our_model_demo.py --plan-only

# Run a short route test
python our_model_demo.py --vehicles 0 --no-yolo --max-seconds 120
```

Both demos use `steer_model.pth` and `steer_normalisation_params.csv` by default.
They accept `--weights` and `--norm` to select alternate tabular-model files.

## Model workflows

There are two distinct dataset/model workflows. The old RGB-only steering
dataset is **not** a substitute for the multimodal dataset: it does not contain
depth, segmentation, LiDAR, radar, or the full state labels required by the
fusion model.

### Existing image-only dataset and PilotNet

With CARLA running, `get_dataset.py` can collect center/left/right RGB images and
steering, throttle, brake, and speed labels:

```powershell
python get_dataset.py --mode steer --samples 3000 --town Town03 --out drive_data
python train_steer.py --data drive_data --epochs 30
```

`train_steer.py` trains the legacy image-based PilotNet and writes
`steer_model.pt` and `pilotnet_model.onnx`. These legacy artifacts are not
included by default; the script can regenerate them. PilotNet is separate from
the current tabular controller and the new multimodal model.

### Multimodal collection

[collect_data.py](./collect_data.py) records synchronous, same-frame sensor
data and Traffic Manager control labels. It saves:

- RGB, metric depth, and semantic-segmentation class images.
- LiDAR points and radar detections (including relative velocity).
- Speed, motion, traffic-light state, and speed limit.
- Steering, throttle, brake, episode, weather, recovery flag, CARLA frame, and
  simulation timestamp.

The recorder varies weather, vehicle and pedestrian counts, and periodically
injects a brief steering disturbance. It then records the Traffic Manager's
post-disturbance recovery controls. It reloads requested towns and restores the
original map, weather, and world settings when it finishes. Collection temporarily
controls the CARLA server; do not run it while another user depends on that
server.

Example recommended collection:

```powershell
python collect_data.py `
  --towns Town01,Town03,Town05 `
  --episodes-per-town 3 `
  --samples-per-episode 1200 `
  --vehicles 20 `
  --walkers 10 `
  --out fusion_data
```

For a small sensor-pipeline check, reduce the town list, episode count,
`--samples-per-episode`, `--vehicles`, and `--walkers`. `--recovery-interval 0`
disables the steering disturbances. `collect_data.py --help` lists all sensor,
timing, and recovery options.

Each run appends samples to `fusion_data/manifest.jsonl` and writes each episode's
sensor files in a uniquely named subdirectory. Do not mix or rename individual
sensor files without updating the manifest.

### Train the multimodal model

After collecting data, train with [train.py](./train.py):

```powershell
python train.py --data fusion_data --epochs 50 --batch-size 8
```

Training holds out complete episodes when possible, normalizes vehicle-state
inputs, and writes the best validation checkpoint to `steer_fusion.pth` by
default. Use `--output <path>` to select another checkpoint path. The model
fuses the three RGB channels, depth, semantic class channel, LiDAR, radar, and
state features to predict steering, throttle, and brake.

### Compare the old and new iterations

After the fusion checkpoint has been trained, compare it with the existing
tabular model on the same held-out episodes:

```powershell
python compare_models.py `
  --data fusion_data `
  --previous-weights steer_model.pth `
  --fusion-weights steer_fusion.pth
```

The script writes `results/fusion_comparison.csv` with per-frame ground truth
and predictions, plus `results/fusion_comparison.json` with MAE, RMSE, bias,
and steering-direction accuracy for steering, throttle, and brake. It requires
the previous model and normalization file, the trained fusion checkpoint, and
the multimodal dataset. Compare metrics on the same held-out data; they are not
a substitute for closed-loop CARLA evaluation.

### Run a trained fusion checkpoint

Once `steer_fusion.pth` exists, pass it to either demo:

```powershell
python live_demo.py --fusion-weights steer_fusion.pth --start-model
python our_model_demo.py --fusion-weights steer_fusion.pth
```

The demos attach synchronized depth, semantic, LiDAR, and radar sensors when
`--fusion-weights` is provided. If the argument is omitted, the demos continue
using the existing tabular `steer_model.pth`. Fusion training and data collection
are user-run workflows; if `steer_fusion.pth` is unavailable, run the training
command above before enabling fusion in a demo.

## Navigation and minimap

The fixed route starts at Town03 spawn point 0 and visits P1–P10 from
`config/checkpoints_town03.json`. The minimap uses a road cache for the active
map; if a cache is missing, the live demo can build it from CARLA's map.

```powershell
# Refresh the active map's road cache while running the demo
python live_demo.py --minimap --refresh-map-cache

# Generate caches and previews for selected maps
python scripts\generate_minimap_caches.py --maps Town01 Town03 Town05_Opt

# Render a minimap without CARLA
python scripts\render_offline_minimap.py
```

Map caches are stored under `config/`; generated previews and runtime outputs
are stored under `results/`.

To inspect or regenerate Town03 checkpoints against a live CARLA map:

```powershell
python generate_checkpoints.py --town Town03 --host localhost --port 2000
```

The canonical checkpoint configuration is
`config/checkpoints_town03.json`.

## Tests and diagnostics

Run the offline test suite (no CARLA server or GPU required):

```powershell
python -m unittest discover -s tests
```

The suite covers minimap rendering and map selection, route assembly and
navigation behavior, and multimodal model input/decoder behavior.

For CARLA-specific diagnostics, run:

```powershell
python verify_carla_minimap.py --host localhost --port 2000 --town Town03
python smoke_test.py
```

These commands require a compatible CARLA Python API and a running server.

## Key files

```text
live_demo.py                  Main live CARLA demo
our_model_demo.py             Fixed Town03 route demo
get_dataset.py                Legacy RGB steering/perception data capture
train_steer.py                Legacy PilotNet training
train_kritty.py               Tabular Kritty model training from drive_data
collect_data.py               Synchronized RGB/depth/semantic/LiDAR/radar capture
train.py                      Multimodal fusion-model training
compare_models.py             Held-out old-vs-new model comparison
fused_control_model.py        Fusion architecture and dataset loader
fused_carla_sensors.py        Synchronized fusion sensors for the demos
kritty_motion_controller.py   Existing tabular model inference
pilotnet.py                   Legacy image-only model
navigation/                   Fixed route and checkpoint helpers
visualization/minimap.py      CARLA road minimap
config/                       Checkpoints, road caches, and settings
tests/                        Offline unit tests
```

## Troubleshooting

| Problem | What to check |
|---|---|
| CARLA connection timeout | Start the simulator; verify `--host`, `--port`, and firewall settings. |
| CARLA Python API import failure | Install the wheel matching both the CARLA server and Python version. |
| Fixed route fails outside Town03 | The current checkpoint route is Town03-specific; use `--no-navigation` in `live_demo.py` for other maps. |
| Sensor frame timeout during collection | Reduce simulator load, check sensor/server compatibility, or increase `--sensor-timeout`. |
| `train.py` cannot find data | Confirm `fusion_data/manifest.jsonl` and the referenced sensor files exist. |
| Fusion demo cannot load checkpoint | Run `train.py` first and pass its output path to `--fusion-weights`. |
| OpenCV window unavailable | Run with `--no-display` in `live_demo.py`; use `--video` if a recording is needed. |
| Checkpoints no longer align with the map | Regenerate `config/checkpoints_town03.json` against the matching CARLA map version. |
