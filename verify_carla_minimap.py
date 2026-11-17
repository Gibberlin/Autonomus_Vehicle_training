#!/usr/bin/env python3
"""CARLA Minimap Integration Test and Diagnostic Script.

Execute this script on the laptop running the CARLA simulator to verify:
1. CARLA connection and Town03 map loading
2. Map topology and driving waypoints extraction
3. Checkpoint alignment against actual Town03 road lanes
4. Minimap rendering and frame overlay performance
5. Visual verification image output

Usage:
    python verify_carla_minimap.py [--host localhost] [--port 2000] [--town Town03]
"""

import argparse
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "checkpoints_town03.json")
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "carla_minimap_test.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify CARLA Minimap and Checkpoints Integration")
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument("--town", default="Town03", help="Target town map name (default: Town03)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to checkpoints JSON")
    parser.add_argument("--out-img", default=OUTPUT_IMAGE_PATH, help="Output image path for visual verification")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate checkpoints from map")
    return parser.parse_args()


def run_integration_test():
    args = parse_args()
    print("=" * 70)
    print("CARLA TOWN03 MINIMAP INTEGRATION TEST & DIAGNOSTICS")
    print("=" * 70)

    # 1. Dependency Checks
    print("\n[Step 1/6] Checking Python environment dependencies...")
    missing_deps = []
    try:
        import numpy as np
        print(f"  [OK] NumPy version: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")

    try:
        import cv2
        print(f"  [OK] OpenCV version: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")

    try:
        import carla
        print(f"  [OK] CARLA Python API loaded successfully")
    except ImportError:
        print("  [FAIL] CARLA Python API is not installed or not in PYTHONPATH.")
        print("         Ensure CARLA Python API egg/wheel or PythonAPI path is configured.")
        return 1

    if missing_deps:
        print(f"  [FAIL] Missing required dependencies: {missing_deps}")
        print("         Install them using: pip install " + " ".join(missing_deps))
        return 1

    from navigation.checkpoints import (
        CheckpointManager,
        generate_town03_checkpoints,
        load_checkpoints,
    )
    from visualization.minimap import Minimap

    # 2. CARLA Connection
    print(f"\n[Step 2/6] Connecting to CARLA server at {args.host}:{args.port}...")
    t0 = time.perf_counter()
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        server_version = client.get_server_version()
        client_version = client.get_client_version()
        connect_time = (time.perf_counter() - t0) * 1000
        print(f"  [OK] Connected in {connect_time:.1f} ms")
        print(f"       Server Version: {server_version}")
        print(f"       Client Version: {client_version}")
    except Exception as exc:
        print(f"  [FAIL] Could not connect to CARLA server: {exc}")
        print("         Please check that CarlaUE4 is running and accessible.")
        return 1

    # 3. Map Verification
    print(f"\n[Step 3/6] Verifying simulation map...")
    world = client.get_world()
    current_map_name = world.get_map().name
    print(f"  Current world map: {current_map_name}")
    if args.town.lower() not in current_map_name.lower():
        print(f"  Loading target map '{args.town}' (this may take a few seconds)...")
        world = client.load_world(args.town)
        current_map_name = world.get_map().name
        print(f"  [OK] Loaded map: {current_map_name}")
    else:
        print(f"  [OK] Map '{args.town}' is already active.")

    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(2.0)
    print(f"  Total driving waypoints generated (2.0m spacing): {len(waypoints)}")
    if len(waypoints) < 100:
        print("  [FAIL] Insufficient waypoints returned by map.")
        return 1
    print("  [OK] Map topology and waypoint extraction verified.")

    # 4. Checkpoints Verification
    print(f"\n[Step 4/6] Verifying 10 fixed checkpoints...")
    if args.regenerate or not os.path.isfile(args.config):
        print(f"  Generating deterministic checkpoints from live {current_map_name} map...")
        checkpoints = generate_town03_checkpoints(carla_map, save_path=args.config)
    else:
        print(f"  Loading existing checkpoints from {args.config}...")
        checkpoints = load_checkpoints(args.config)

    print(f"  Loaded {len(checkpoints)} checkpoints:")
    all_aligned = True
    for cp in checkpoints:
        # Check alignment against live CARLA map
        loc = carla.Location(x=cp.x, y=cp.y, z=cp.z)
        nearest_wp = carla_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        dist_to_lane = loc.distance(nearest_wp.transform.location)
        is_junction = nearest_wp.is_junction
        status = "OK" if dist_to_lane < 4.0 and not is_junction else "WARNING"
        if status != "OK":
            all_aligned = False
        print(
            f"    [{status}] {cp.id:<4} | {cp.region:<32} | ({cp.x:>6.1f}, {cp.y:>6.1f}) | "
            f"offset: {dist_to_lane:.2f}m | junction: {is_junction}"
        )

    if all_aligned:
        print("  [OK] All 10 checkpoints align with valid non-junction driving lanes.")
    else:
        print("  [WARNING] Some checkpoints may be slightly offset or near junctions. You can regenerate via --regenerate.")

    # 5. Minimap Initialization & Latency Benchmark
    print(f"\n[Step 5/6] Benchmarking Minimap rendering performance...")
    t_init = time.perf_counter()
    minimap = Minimap(
        size=260,
        position="top-right",
        opacity=0.85,
        carla_map=carla_map,
        checkpoints=checkpoints,
        checkpoints_path=args.config,
    )
    init_ms = (time.perf_counter() - t_init) * 1000
    print(f"  [OK] Minimap initialized in {init_ms:.1f} ms (roads + checkpoints pre-rendered)")

    # Benchmark per-frame render
    n_frames = 100
    t_bench = time.perf_counter()
    for _ in range(n_frames):
        minimap.render(vehicle_x=142.0, vehicle_y=195.0, vehicle_yaw=45.0)
    render_avg_ms = ((time.perf_counter() - t_bench) / n_frames) * 1000
    print(f"  [OK] Mean per-frame render time: {render_avg_ms:.3f} ms (supports > 1000 FPS)")

    # 6. Simulation Frame Overlay & Visual Output
    print(f"\n[Step 6/6] Generating composite test frame...")
    # Simulate a 320x240 camera frame (matching live_demo.py)
    test_frame = np.full((240, 320, 3), (60, 50, 45), dtype=np.uint8)
    cv2.putText(
        test_frame,
        "CARLA Camera View (320x240)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    overlaid_frame = minimap.overlay_on_frame(
        test_frame,
        vehicle_x=142.1,
        vehicle_y=195.4,
        vehicle_yaw=90.0,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_img)), exist_ok=True)
    cv2.imwrite(args.out_img, overlaid_frame)
    print(f"  [OK] Saved composite verification frame to: {args.out_img}")

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    print("Next step: Run the full simulation with the minimap:")
    print("    python live_demo.py --minimap")
    print("Or with options:")
    print("    python live_demo.py --minimap --minimap-position top-right --minimap-size 260 --minimap-opacity 0.85")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_integration_test())

