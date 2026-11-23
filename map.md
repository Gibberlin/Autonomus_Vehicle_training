# Project Map

This map starts at the repository root. The root contains both the project code
and a CARLA simulator distribution.

```text
.
|-- map.md
|-- README.md
|-- CarlaUE4.exe
|-- CarlaUE4/                 CARLA simulator project and assets
|-- Engine/                   Unreal Engine files used by CARLA
|-- Co-Simulation/            CARLA co-simulation integrations
|-- HDMaps/                   High-definition map assets
|-- PythonAPI/                CARLA Python API, examples, and client tools
|-- LocalMap/                 Custom Guwahati map files and loader
|-- config/                   Routes and map/road caches
|-- navigation/               Route planning and navigation helpers
|-- visualization/            Minimap rendering
|-- scripts/                  Map, report, and offline rendering utilities
|-- tests/                    Offline unit tests
|-- kritty/                   Cooperative-perception and KITTI control prototypes
|-- dataset/                  YOLO images, labels, metadata, and previews
|-- drive_data/               RGB steering data and labels
|-- fusion_data/              Multimodal CARLA sensor data and manifest
|-- frames/                   Frames used by detection benchmarks
|-- clip_frames/              Frames captured for short video clips
|-- results/                  Generated training, demo, and visualization output
|-- report/                   Generated report figures
|-- carla-env/                Local Python virtual environment
|-- .cache/                   Local tool/model caches
|-- .vscode/                  Workspace editor settings
|-- __pycache__/              Python bytecode cache
|-- .git/                     Git repository metadata
```

## Root-level files

### Demos and navigation

- `live_demo.py` - Main CARLA driving demo; includes route navigation and
  optional YOLO detection and fusion-model sensors.
- `our_model_demo.py` - Fixed Town03 route demo using the steering controller.
- `generate_checkpoints.py` - Generate route checkpoints from a CARLA map.
- `verify_carla_minimap.py` - Check CARLA/map connectivity and minimap setup.
- `smoke_test.py` - Basic CARLA connection and camera smoke test.
- `navigation/` - Route assembly, checkpoint handling, and fixed-route logic.
- `config/` - Town03 route checkpoints and cached road geometry for minimaps.
- `visualization/` - Minimap drawing and related visualization helpers.

### Data capture and computer vision

- `collect_data.py` - Synchronized multimodal capture: RGB, depth, semantic
  segmentation, LiDAR, radar, vehicle state, and control labels.
- `get_dataset.py` - Legacy center/left/right RGB capture for PilotNet training.
- `record_drive.py` - Older three-camera RGB driving-data recorder; writes to
  `drive_data/`.
- `record.py` - CARLA scene/image and actor-metadata capture for object labels.
- `record_clip.py` - Capture a short CARLA camera sequence into `clip_frames/`.
- `make_labels.py` - Convert captured object metadata into YOLO labels.
- `detect.py` - Run YOLO over `frames/` and save inference benchmark CSVs.
- `render_clip.py` - Run YOLO over captured clip frames and render a detection
  video.
- `dataset.yaml` - YOLO dataset configuration.
- `yolov8n.pt` - YOLOv8 nano detector weights.

### Models and training

- `fused_control_model.py` - Multimodal model architecture and dataset loading.
- `fused_carla_sensors.py` - Synchronized depth, semantic, LiDAR, and radar
  sensors for the demos.
- `train.py` - Train the multimodal fusion controller from `fusion_data/`.
- `compare_models.py` - Compare the tabular controller with a trained fusion
  controller on the same held-out data.
- `kritty_motion_controller.py` - Load and run the current tabular controller.
- `train_kritty.py` - Retrain the tabular controller from `drive_data/`.
- `pilotnet.py` - Legacy image-only PilotNet architecture.
- `train_steer.py` - Legacy PilotNet training script.
- `steer_model.pth` - Current tabular steering model checkpoint.
- `steer_model_ts.pt` - TorchScript export of the tabular controller.
- `steer_model.onnx` - ONNX export of the tabular controller.
- `steer_fusion.pth` - Multimodal fusion controller checkpoint.
- `steer_normalisation_params.csv` - Input normalization values for the tabular
  controller.
- `steer_target_scales.csv` - Output scaling values used by the tabular model.

### Utilities and project metadata

- `render_clip.py`, `record_clip.py` - Clip capture and rendering utilities.
- `requirements.txt` - Main Python dependencies.
- `Dockerfile` - Container build configuration.
- `.gitignore` - Excludes local environments, datasets, caches, and generated
  files while retaining the main model artifacts.
- `README.md` - Setup, demos, data collection, training, and troubleshooting.
- `CHANGELOG` - Project change history.
- `LICENSE` - Project license.
- `baseline_cpu_fp32.csv`, `baseline_gpu_fp16.csv`,
  `baseline_gpu_fp32.csv` - YOLO inference benchmark results.
- `detections_gpu.mp4` - Generated detection video.
- `crop_check.png` - Generated image/crop check.
- `progress.pdf` - Project progress report.

## Important folders

- `kritty/carla_cooperative_perception.py` - CARLA multi-vehicle cooperative
  perception prototype.
- `kritty/kitti_motion_control/` - Separate KITTI-based driving prototype.
  `train_model.py` trains it; `predict_control.py` uses the current root
  steering model by default; `processed_data/` contains its prepared datasets.
  `model/` is reserved for generated model outputs.
- `dataset/` - Object-detection data, organized into `images/`, `meta/`,
  `labels/`, and `preview/`.
- `drive_data/` - PilotNet/tabular steering capture: images and `labels.csv`.
- `fusion_data/` - Multimodal capture episodes plus `manifest.jsonl`.
- `frames/` and `clip_frames/` - Local image collections used by detection and
  clip-processing tools.
- `results/` and `report/` - Generated model/demo outputs and report figures.
- `scripts/` - Utilities for map-cache generation, offline minimap rendering,
  and report generation.
- `tests/` - Offline tests for navigation, minimaps, and the fusion model.
- `LocalMap/` - Guwahati OpenStreetMap/OpenDRIVE files and map loader.
- `PythonAPI/`, `CarlaUE4/`, `Engine/`, `Co-Simulation/`, and `HDMaps/` - CARLA
  simulator distribution components, rather than application model code.
