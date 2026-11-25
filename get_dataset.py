#!/usr/bin/env python3
"""CARLA Dataset Generation Tool.

Automatically launches the CARLA simulator (if not already running) and captures
high-quality, multi-camera autonomous driving training data (for PilotNet steering)
and/or perception data (for YOLO object detection).

Usage Examples:
    # 1. Capture 3,000 driving samples (~9,000 images, center/left/right) for PilotNet:
    python get_dataset.py --mode steer --samples 3000 --town Town03

    # 2. Capture YOLO perception dataset with vehicles, pedestrians, and weather:
    python get_dataset.py --mode perception --samples 1000 --town Town03

    # 3. Use an existing CARLA server running on port 2000:
    python get_dataset.py --no-launch --samples 2000

    # 4. Launch CARLA headless on low quality for maximum capture FPS:
    python get_dataset.py --headless --quality Low --samples 3000
"""

import argparse
import atexit
import csv
import json
import os
import queue
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

# Add project root and CARLA Python API to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_PATH = PROJECT_ROOT / "PythonAPI" / "carla"
if API_PATH.is_dir() and str(API_PATH) not in sys.path:
    sys.path.insert(0, str(API_PATH))


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if CARLA RPC port is listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def find_carla_executable(custom_path: str | None = None) -> Path | None:
    """Find CarlaUE4 executable on the host system."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    # Standard workspace locations
    candidates.extend([
        PROJECT_ROOT / "CarlaUE4.exe",
        PROJECT_ROOT / "CarlaUE4.sh",
        PROJECT_ROOT / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4-Win64-Shipping.exe",
    ])

    # Environment variables
    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        candidates.extend([
            Path(carla_root) / "CarlaUE4.exe",
            Path(carla_root) / "CarlaUE4.sh",
        ])

    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def launch_carla_server(
    exe_path: Path,
    port: int = 2000,
    quality: str = "Low",
    headless: bool = False,
    timeout: float = 60.0,
) -> subprocess.Popen:
    """Launch CarlaUE4 process and wait until the RPC port is reachable."""
    print(f"[CARLA Launcher] Launching: {exe_path}")
    cmd = [
        str(exe_path),
        "-carla-server",
        f"-carla-rpc-port={port}",
        f"-quality-level={quality}",
    ]
    if headless:
        cmd.append("-RenderOffScreen")
    else:
        cmd.extend(["-windowed", "-ResX=800", "-ResY=600"])

    print(f"[CARLA Launcher] Command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(exe_path.parent),
    )

    # Wait for CARLA server to accept connections
    print(f"[CARLA Launcher] Waiting for CARLA server on port {port} (timeout {timeout:.0f}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"CARLA process terminated unexpectedly with code {proc.returncode}")
        if is_port_open("localhost", port, timeout=0.8):
            print(f"[CARLA Launcher] CARLA server is online and accepting connections! (took {time.time() - start_time:.1f}s)")
            return proc
        time.sleep(1.0)

    proc.terminate()
    raise TimeoutError(f"CARLA server failed to open port {port} within {timeout} seconds.")


def safe_cleanup_actors(client, actors):
    """Safely stop sensors and destroy actors in batch."""
    if not actors:
        return
    alive = [a for a in actors if a is not None and a.is_alive]
    for a in alive:
        try:
            if hasattr(a, "stop"):
                a.stop()
        except Exception:
            pass
    try:
        import carla
        client.apply_batch([carla.command.DestroyActor(x.id) for x in alive])
    except Exception as e:
        print(f"[Cleanup Warning] Actor destruction batch encountered: {e}")


def capture_steer_dataset(args, client, world):
    """Capture 3-camera driving dataset (center, left, right) with telemetry labels."""
    import cv2
    import numpy as np
    import carla

    out_dir = Path(args.out)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"

    print(f"\n[Steering Dataset] Output directory: {out_dir}")
    print(f"[Steering Dataset] Target samples: {args.samples} (x3 cameras = {args.samples * 3} images)")

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points found on the current map!")
    random.shuffle(spawns)

    # Synchronous simulation settings
    sim_settings = world.get_settings()
    sim_settings.synchronous_mode = True
    sim_settings.fixed_delta_seconds = args.fixed_delta
    world.apply_settings(sim_settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(3.0)

    actors = []
    cams = []
    queues = {}

    try:
        # Spawn Ego vehicle
        ego_bp = bp.find("vehicle.tesla.model3")
        ego = world.try_spawn_actor(ego_bp, spawns.pop())
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle at initial spawn point")
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)

        # Spawn NPC traffic
        car_bps = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        npc_count = 0
        for sp in spawns[:args.vehicles]:
            v = world.try_spawn_actor(random.choice(car_bps), sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)
                npc_count += 1
        print(f"[Traffic] Spawned ego vehicle + {npc_count} NPC traffic vehicles")

        # Multi-camera setup: center, left (-0.5m), right (+0.5m)
        w, h = 320, 240
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(w))
        cam_bp.set_attribute("image_size_y", str(h))
        cam_bp.set_attribute("fov", "90")

        cam_offsets = {
            "center": carla.Location(x=1.5, y=0.0, z=2.0),
            "left":   carla.Location(x=1.5, y=-0.5, z=2.0),
            "right":  carla.Location(x=1.5, y=0.5, z=2.0),
        }

        for name, loc in cam_offsets.items():
            cam_actor = world.spawn_actor(cam_bp, carla.Transform(loc), attach_to=ego)
            queues[name] = queue.Queue()
            cam_actor.listen(queues[name].put)
            cams.append(cam_actor)
            actors.append(cam_actor)

        # Warmup ticks to let traffic and physics stabilize
        print("[Warmup] Stabilizing simulation physics and traffic...")
        for _ in range(60):
            world.tick()
            for q in queues.values():
                q.get(timeout=10.0)

        # Open CSV file
        file_mode = "a" if args.append and csv_path.is_file() else "w"
        write_header = not (args.append and csv_path.is_file())

        saved = 0
        tick = 0
        left_turns = 0
        right_turns = 0
        straights = 0
        t0 = time.time()

        with open(csv_path, file_mode, newline="") as fh:
            wr = csv.writer(fh)
            if write_header:
                wr.writerow(["frame", "cam", "file", "steer", "throttle", "brake", "speed"])

            print(f"[Recording] Capturing samples to {csv_path.name}...")
            while saved < args.samples:
                world.tick()
                tick += 1

                imgs = {name: q.get(timeout=10.0) for name, q in queues.items()}
                if tick % args.save_every != 0:
                    continue

                ctrl = ego.get_control()
                vel = ego.get_velocity()
                speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5  # km/h

                # Skip excessive stopped frames (keep only 10% when stationary)
                if speed < 1.0 and random.random() > 0.10:
                    continue

                steer_val = round(ctrl.steer, 4)
                if steer_val < -0.05:
                    left_turns += 1
                elif steer_val > 0.05:
                    right_turns += 1
                else:
                    straights += 1

                for name, im in imgs.items():
                    arr = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
                    fn = f"{im.frame:07d}_{name}.jpg"
                    cv2.imwrite(str(img_dir / fn), arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    wr.writerow([
                        im.frame,
                        name,
                        fn,
                        steer_val,
                        round(ctrl.throttle, 3),
                        round(ctrl.brake, 3),
                        round(speed, 2),
                    ])

                saved += 1
                if saved % 250 == 0 or saved == args.samples:
                    elapsed = time.time() - t0
                    fps = saved / elapsed if elapsed > 0 else 0
                    print(
                        f"  [Progress] {saved:5d}/{args.samples} samples ({saved * 3:6d} imgs) | "
                        f"{fps:.1f} Hz | Left: {left_turns:4d} | Straight: {straights:4d} | Right: {right_turns:4d}"
                    )

        total_time = time.time() - t0
        print("\n" + "=" * 60)
        print("[Steering Dataset Complete]")
        print(f"  Total Samples: {saved} ({saved * 3} multi-cam images)")
        print(f"  Distribution: Left={left_turns} ({left_turns/saved:.1%}), Straight={straights} ({straights/saved:.1%}), Right={right_turns} ({right_turns/saved:.1%})")
        print(f"  Elapsed Time: {total_time:.1f}s ({saved / total_time:.1f} samples/s)")
        print(f"  Saved to: {csv_path}")
        print("=" * 60)

    finally:
        print("[Cleanup] Stopping cameras and destroying simulation actors...")
        safe_cleanup_actors(client, actors)
        sim_settings.synchronous_mode = False
        world.apply_settings(sim_settings)
        tm.set_synchronous_mode(False)


def capture_perception_dataset(args, client, world):
    """Capture perception dataset with bounding box labels across diverse weather."""
    import carla

    out_dir = Path("dataset")
    img_dir = out_dir / "images"
    meta_dir = out_dir / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    weathers = {
        "clear": carla.WeatherParameters.ClearNoon,
        "rain":  carla.WeatherParameters.HardRainNoon,
        "fog":   carla.WeatherParameters(cloudiness=80, fog_density=60, fog_distance=10, sun_altitude_angle=45),
        "night": carla.WeatherParameters.ClearNight,
    }

    print(f"\n[Perception Dataset] Output directory: {out_dir}")
    print(f"[Perception Dataset] Weathers: {list(weathers.keys())}")

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    sim_settings = world.get_settings()
    sim_settings.synchronous_mode = True
    sim_settings.fixed_delta_seconds = args.fixed_delta
    world.apply_settings(sim_settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)

    actors, walkers, controllers = [], [], []

    try:
        # Spawn Ego vehicle
        ego = world.try_spawn_actor(bp.find("vehicle.tesla.model3"), spawns.pop())
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)

        # Spawn NPC vehicles
        car_bps = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        for sp in spawns[:args.vehicles]:
            v = world.try_spawn_actor(random.choice(car_bps), sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)

        # Spawn Pedestrians
        walker_ctrl_bp = bp.find("controller.ai.walker")
        walker_bps = bp.filter("walker.pedestrian.*")
        for _ in range(args.walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                w = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
                if w:
                    walkers.append(w)
                    actors.append(w)

        world.tick()
        for w in walkers:
            c = world.spawn_actor(walker_ctrl_bp, carla.Transform(), attach_to=w)
            controllers.append(c)
            actors.append(c)

        world.tick()
        for c in controllers:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.2 + random.random() * 0.8)

        # Camera setup (640x480 for object detection)
        w, h, fov = 640, 480, 90
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(w))
        cam_bp.set_attribute("image_size_y", str(h))
        cam_bp.set_attribute("fov", str(fov))
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.0)), attach_to=ego)
        actors.append(cam)

        q = queue.Queue()
        cam.listen(q.put)

        targets = [
            a for a in world.get_actors()
            if a.type_id.startswith(("vehicle.", "walker.pedestrian")) and a.id != ego.id
        ]

        frames_per_weather = max(1, args.samples // len(weathers))
        max_dist = 60.0
        total_saved = 0

        for wname, weather in weathers.items():
            print(f"[Weather Change] Switching to: {wname}")
            world.set_weather(weather)
            for _ in range(40):
                world.tick()
                q.get(timeout=10.0)

            for i in range(frames_per_weather):
                for _ in range(args.save_every):
                    world.tick()
                    img = q.get(timeout=10.0)

                name = f"{wname}_{i:04d}"
                img.save_to_disk(str(img_dir / f"{name}.png"))

                ego_loc = ego.get_location()
                objs = []
                for a in targets:
                    if not a.is_alive or a.get_location().distance(ego_loc) > max_dist:
                        continue
                    verts = a.bounding_box.get_world_vertices(a.get_transform())
                    objs.append({
                        "id": a.id,
                        "cls": "pedestrian" if a.type_id.startswith("walker") else "vehicle",
                        "verts": [[v.x, v.y, v.z] for v in verts],
                    })

                meta = {
                    "image": f"{name}.png",
                    "width": w,
                    "height": h,
                    "fov": fov,
                    "world_to_cam": cam.get_transform().get_inverse_matrix(),
                    "objects": objs,
                }
                with open(meta_dir / f"{name}.json", "w") as mfh:
                    json.dump(meta, mfh)

                total_saved += 1
                if total_saved % 50 == 0:
                    print(f"  [Perception] Saved {total_saved} frames with 3D bounding boxes")

        print(f"[Perception Dataset Complete] Saved {total_saved} frames to {out_dir}")

    finally:
        print("[Cleanup] Cleaning up perception actors...")
        safe_cleanup_actors(client, actors)
        sim_settings.synchronous_mode = False
        world.apply_settings(sim_settings)
        tm.set_synchronous_mode(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch CARLA and capture autonomous vehicle training data"
    )
    parser.add_argument("--mode", choices=("steer", "perception", "all"), default="steer",
                        help="dataset capture mode (steer=PilotNet driving, perception=YOLO bboxes, all=both)")
    parser.add_argument("--samples", type=int, default=3000,
                        help="number of data samples to record (default: 3000)")
    parser.add_argument("--town", default="Town03",
                        help="CARLA map town name to load (default: Town03)")
    parser.add_argument("--out", default="drive_data",
                        help="output directory for steering dataset (default: drive_data)")
    parser.add_argument("--vehicles", type=int, default=25,
                        help="number of NPC traffic vehicles to spawn (default: 25)")
    parser.add_argument("--walkers", type=int, default=15,
                        help="number of pedestrians for perception mode (default: 15)")
    parser.add_argument("--host", default="localhost",
                        help="CARLA host address (default: localhost)")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA RPC port (default: 2000)")
    parser.add_argument("--tm-port", type=int, default=8000,
                        help="Traffic Manager port (default: 8000)")
    parser.add_argument("--fixed-delta", type=float, default=0.05,
                        help="simulation step delta seconds (default: 0.05, 20 Hz)")
    parser.add_argument("--save-every", type=int, default=2,
                        help="ticks between saved frames (default: 2 -> 10 Hz recording)")
    parser.add_argument("--append", action="store_true",
                        help="append to existing labels.csv instead of overwriting")
    parser.add_argument("--no-launch", action="store_true",
                        help="do not attempt to launch CarlaUE4.exe; connect to an existing server")
    parser.add_argument("--carla-path", default=None,
                        help="explicit path to CarlaUE4 executable")
    parser.add_argument("--headless", action="store_true",
                        help="launch CARLA server in headless offscreen mode (-RenderOffScreen)")
    parser.add_argument("--quality", choices=("Low", "Epic"), default="Low",
                        help="CARLA graphics quality level (default: Low for high FPS)")
    parser.add_argument("--keep-carla", action="store_true",
                        help="keep CARLA server running after dataset capture completes")
    return parser.parse_args()


def main():
    args = parse_args()
    carla_proc = None

    # Step 1: Ensure CARLA server is running
    is_running = is_port_open(args.host, args.port, timeout=1.0)
    if is_running:
        print(f"[Connection] CARLA server already running at {args.host}:{args.port}")
    elif args.no_launch:
        print(f"[Error] CARLA server is not running at {args.host}:{args.port} and --no-launch was specified.", file=sys.stderr)
        return 1
    else:
        carla_exe = find_carla_executable(args.carla_path)
        if not carla_exe:
            print(f"[Error] CarlaUE4 executable not found in workspace or paths. Please pass --carla-path.", file=sys.stderr)
            return 1

        carla_proc = launch_carla_server(
            exe_path=carla_exe,
            port=args.port,
            quality=args.quality,
            headless=args.headless,
        )

        def cleanup_process():
            if carla_proc and carla_proc.poll() is None and not args.keep_carla:
                print("[Process Cleanup] Terminating CARLA server...")
                carla_proc.terminate()
                try:
                    carla_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    carla_proc.kill()

        atexit.register(cleanup_process)

    # Step 2: Connect to CARLA Python API
    try:
        import carla
    except ImportError as e:
        print(f"[Error] CARLA Python API could not be imported: {e}", file=sys.stderr)
        print("Please ensure you are using the 'carla-env' virtual environment.", file=sys.stderr)
        return 1

    print(f"[Connection] Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    # Step 3: Load requested map
    current_map = client.get_world().get_map().name
    target_town = args.town
    if target_town:
        # Match against available CARLA maps (resolves e.g. "Town10" -> "Town10HD" or "Town10HD_Opt")
        try:
            available_maps = client.get_available_maps()
            target_clean = target_town.strip().casefold()
            matched_map = None
            for m in available_maps:
                short = m.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                if short.casefold() == target_clean:
                    matched_map = short
                    break
            if not matched_map:
                for m in available_maps:
                    short = m.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                    if short.casefold().startswith(target_clean) or target_clean in short.casefold():
                        matched_map = short
                        break
            if matched_map:
                target_town = matched_map
        except Exception:
            pass

    if target_town and not current_map.endswith(target_town):
        print(f"[Map] Switching map from {current_map} to {target_town}...")
        world = client.load_world(target_town)
    else:
        print(f"[Map] Using map: {current_map}")
        world = client.get_world()

    # Step 4: Execute requested dataset capture mode
    try:
        if args.mode in ("steer", "all"):
            capture_steer_dataset(args, client, world)
        if args.mode in ("perception", "all"):
            capture_perception_dataset(args, client, world)
    except KeyboardInterrupt:
        print("\n[Interrupted] Dataset recording aborted by user.")
    except Exception as e:
        print(f"\n[Error] Exception during dataset recording: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print("\n[Done] get_dataset.py completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
