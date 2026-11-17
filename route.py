import argparse
import csv
import os
import queue
import random
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path


CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_FOV = 90.0
CAMERA_HEIGHT_M = 2.0
ROUTE_TOWN = "Town03"
ROUTE_START_SPAWN_INDEX = 0
ROUTE_DISTANCE_M = 5_000.0
ROUTE_RESOLUTION_M = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive a fixed 5 km Town03 route using PilotNet steering"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--weights", default="steer_model.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--vehicles", type=int, default=25, help="number of NPC vehicles")
    parser.add_argument("--target-speed", type=float, default=25.0, help="speed in km/h")
    parser.add_argument("--fixed-delta", type=float, default=0.05, help="simulation step in seconds")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="optional test-run time limit")
    parser.add_argument("--video", default=None, help="optional output video path")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--no-yolo", action="store_true", help="disable object detection")
    parser.add_argument("--plan-only", action="store_true", help="plan and report the fixed route without driving")
    return parser.parse_args()


def build_fixed_route(world_map):
    api_path = Path(__file__).resolve().parent / "PythonAPI" / "carla"
    if not api_path.is_dir():
        raise RuntimeError(f"CARLA navigation API not found: {api_path}")
    if str(api_path) not in sys.path:
        sys.path.insert(0, str(api_path))

    import networkx as nx
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    spawn_points = world_map.get_spawn_points()
    if ROUTE_START_SPAWN_INDEX >= len(spawn_points):
        raise RuntimeError(
            f"{ROUTE_TOWN} has {len(spawn_points)} spawn points; "
            f"fixed start index {ROUTE_START_SPAWN_INDEX} is invalid"
        )

    planner = GlobalRoutePlanner(world_map, ROUTE_RESOLUTION_M)
    remaining = set(range(len(spawn_points)))
    remaining.remove(ROUTE_START_SPAWN_INDEX)
    current_location = spawn_points[ROUTE_START_SPAWN_INDEX].location
    route = []
    route_distance = 0.0
    previous_location = None

    while remaining and route_distance < ROUTE_DISTANCE_M:
        destination_index = max(
            remaining,
            key=lambda index: (
                current_location.distance(spawn_points[index].location),
                -index,
            ),
        )
        destination = spawn_points[destination_index].location
        try:
            segment = planner.trace_route(current_location, destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            remaining.remove(destination_index)
            continue

        for waypoint, road_option in segment:
            location = waypoint.transform.location
            if previous_location is not None:
                step_distance = previous_location.distance(location)
                if step_distance < 0.05:
                    continue
                if route_distance + step_distance > ROUTE_DISTANCE_M:
                    return route, route_distance, planner, spawn_points
                route_distance += step_distance
            route.append((waypoint, road_option))
            previous_location = location

        current_location = destination
        remaining.remove(destination_index)

    if route_distance < ROUTE_DISTANCE_M - ROUTE_RESOLUTION_M:
        raise RuntimeError(
            f"Could only plan {route_distance:.1f} m on {ROUTE_TOWN}; "
            f"the fixed route requires {ROUTE_DISTANCE_M:.0f} m"
        )
    return route, route_distance, planner, spawn_points


def enqueue_latest(frame_queue, image):
    try:
        frame_queue.put_nowait(image)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put_nowait(image)


def safe_destroy(client, carla, actors):
    alive_actors = [actor for actor in actors if actor is not None]
    if not alive_actors:
        return
    try:
        responses = client.apply_batch(
            [carla.command.DestroyActor(actor.id) for actor in alive_actors]
        )
        if responses is not None:
            for response in responses:
                if response.error:
                    print(f"[cleanup] Actor destruction failed: {response.error}")
    except Exception as exc:
        print(f"[cleanup] Could not destroy all CARLA actors: {exc}")


def speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5


def main():
    args = parse_args()
    if args.vehicles < 0:
        raise ValueError("--vehicles cannot be negative")
    if args.target_speed < 0:
        raise ValueError("--target-speed cannot be negative")
    if args.fixed_delta <= 0:
        raise ValueError("--fixed-delta must be greater than zero")
    if args.max_seconds < 0:
        raise ValueError("--max-seconds cannot be negative")

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is unavailable; run with CARLA's Python environment"
        ) from exc

    client = world = traffic_manager = ego = None
    original_settings = None
    sensors, actors = [], []
    writer = telemetry = None
    cv2 = np = torch = None
    yolo = None
    status, error_text = "ok", ""
    sim_time = driven_distance = model_time = 0.0
    planned_distance = 0.0
    frames = 0
    collision_count = lane_invasion_count = 0
    stats_lock = threading.Lock()
    previous_location = None
    model_steer_previous = 0.0
    mode = {"name": "model"}
    yolo_frame_number = -1
    yolo_ms = 0.0
    yolo_boxes = None
    yolo_classes = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"route_{stamp}.csv")

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.load_world(ROUTE_TOWN)
        original_settings = world.get_settings()

        route, planned_distance, global_planner, spawn_points = build_fixed_route(world.get_map())
        if not route:
            raise RuntimeError("The fixed route planner returned an empty route")
        print(
            f"[route] {ROUTE_TOWN}, start spawn {ROUTE_START_SPAWN_INDEX}, "
            f"{planned_distance / 1000:.3f} km, {len(route)} waypoints, "
            f"destination=({route[-1][0].transform.location.x:.1f}, "
            f"{route[-1][0].transform.location.y:.1f})"
        )
        if args.plan_only:
            return 0

        try:
            import cv2
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("This demo requires OpenCV and PyTorch") from exc
        try:
            from pilotnet import PilotNet, preprocess
            from agents.navigation.basic_agent import BasicAgent
            from live_demo import YoloWorker, find_lead
        except ImportError as exc:
            raise RuntimeError(
                "pilotnet.py, live_demo.py, and CARLA's navigation dependencies must be available"
            ) from exc

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"PilotNet weights not found: {args.weights}")

        model = PilotNet().to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()

        yolo_device = args.yolo_device or ("0" if device == "cuda" else "cpu")
        if not args.no_yolo:
            yolo = YoloWorker(args.yolo_weights, yolo_device, args.yolo_half)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_synchronous_mode(True)

        blueprints = world.get_blueprint_library()
        ego_blueprint = blueprints.find("vehicle.tesla.model3")
        ego = world.try_spawn_actor(
            ego_blueprint, spawn_points[ROUTE_START_SPAWN_INDEX]
        )
        if ego is None:
            raise RuntimeError(
                f"Could not spawn at fixed {ROUTE_TOWN} spawn point "
                f"{ROUTE_START_SPAWN_INDEX}; ensure it is not occupied"
            )
        actors.append(ego)

        vehicle_blueprints = [
            blueprint
            for blueprint in blueprints.filter("vehicle.*")
            if int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        other_spawns = [
            spawn for index, spawn in enumerate(spawn_points)
            if index != ROUTE_START_SPAWN_INDEX
        ]
        random.Random(2026).shuffle(other_spawns)
        for spawn in other_spawns[:args.vehicles]:
            npc = world.try_spawn_actor(random.choice(vehicle_blueprints), spawn)
            if npc is not None:
                npc.set_autopilot(True, traffic_manager.get_port())
                actors.append(npc)
        print(f"[traffic] spawned {len(actors) - 1} of {args.vehicles} requested NPC vehicles")

        agent = BasicAgent(
            ego,
            target_speed=args.target_speed,
            opt_dict={"sampling_resolution": ROUTE_RESOLUTION_M},
            map_inst=world.get_map(),
            grp_inst=global_planner,
        )
        agent.set_global_plan(route, stop_waypoint_creation=True)

        camera_queue = queue.Queue(maxsize=1)
        camera_blueprint = blueprints.find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_blueprint.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_blueprint.set_attribute("fov", str(CAMERA_FOV))
        camera = world.spawn_actor(
            camera_blueprint,
            carla.Transform(carla.Location(x=1.5, z=CAMERA_HEIGHT_M)),
            attach_to=ego,
        )
        sensors.append(camera)
        camera.listen(lambda image: enqueue_latest(camera_queue, image))

        collision_blueprint = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_blueprint, carla.Transform(), attach_to=ego
        )
        sensors.append(collision_sensor)
        lane_blueprint = blueprints.find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(
            lane_blueprint, carla.Transform(), attach_to=ego
        )
        sensors.append(lane_sensor)

        def increment_stat(_event, key):
            _ = _event
            nonlocal collision_count, lane_invasion_count
            with stats_lock:
                if key == "collision":
                    collision_count += 1
                else:
                    lane_invasion_count += 1

        collision_sensor.listen(lambda event: increment_stat(event, "collision"))
        lane_sensor.listen(lambda event: increment_stat(event, "lane_invasion"))

        telemetry = open(csv_path, "w", newline="", encoding="utf-8")
        telemetry_writer = csv.writer(telemetry)
        telemetry_writer.writerow([
            "sim_time_s", "world_frame", "mode", "speed_kmh", "model_steer",
            "route_steer", "command_steer", "throttle", "brake", "distance_m",
            "route_progress_m", "yolo_ms", "yolo_age_frames", "detections", "lead_distance_m",
            "collisions", "lane_invasions",
        ])

        print("[ready] Fixed 5 km PilotNet point-to-point route")
        print("[keys] m: PilotNet steering | r: route-controller baseline | q / Esc: quit")
        print("[mode] PilotNet steering active")
        print("[control] PilotNet supplies all steering; the route controller supplies speed and hazard braking")
        print("[note] PilotNet sees the camera, not the destination; this test may not reach the finish")
        if yolo is None:
            print("[yolo] disabled")
        else:
            print(f"[yolo] recognizing objects with {args.yolo_weights} on {yolo_device}")

        while True:
            world_frame = world.tick()
            image = camera_queue.get(timeout=10.0)
            frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3].copy()

            if yolo is not None:
                camera_frame = int(image.frame)
                yolo.submit(camera_frame, frame)
                latest = yolo.latest()
                if latest is not None:
                    yolo_frame_number, yolo_ms, yolo_boxes, yolo_classes = latest

            model_input = preprocess(frame)
            model_tensor = (
                torch.from_numpy(model_input).unsqueeze(0).to(device).float() / 255.0
            )
            with torch.no_grad():
                model_steer_raw = float(model(model_tensor).item())
            if device == "cuda":
                torch.cuda.synchronize()
            model_steer = float(
                max(-1.0, min(1.0, 0.75 * model_steer_previous + 0.25 * model_steer_raw))
            )
            model_steer_previous = model_steer
            route_control = agent.run_step()

            route_steer = float(route_control.steer)
            if mode["name"] == "model":
                command_steer = model_steer
            else:
                command_steer = route_steer

            throttle = float(route_control.throttle)
            brake = float(route_control.brake)
            control = carla.VehicleControl(
                throttle=throttle,
                steer=command_steer,
                brake=brake,
            )
            if mode["name"] != "route":
                model_time += args.fixed_delta
            ego.apply_control(control)

            remaining_waypoints = len(agent.get_local_planner().get_plan())
            route_progress = planned_distance * (
                1.0 - remaining_waypoints / max(len(route), 1)
            )
            route_progress = max(0.0, min(planned_distance, route_progress))
            lead = (
                find_lead(yolo_boxes, yolo_classes, frame.shape[1], frame.shape[0])
                if yolo is not None and yolo_boxes is not None
                else None
            )
            yolo_age_frames = (
                max(0, int(image.frame) - yolo_frame_number)
                if yolo_frame_number >= 0
                else 0
            )
            speed = speed_kmh(ego)
            location = ego.get_location()
            if previous_location is not None:
                driven_distance += location.distance(previous_location)
            previous_location = location
            sim_time += args.fixed_delta
            frames += 1

            with stats_lock:
                current_collisions = collision_count
                current_invasions = lane_invasion_count
            telemetry_writer.writerow([
                round(sim_time, 2),
                int(world_frame),
                mode["name"],
                round(speed, 3),
                round(model_steer, 4),
                round(float(route_control.steer), 4),
                round(command_steer, 4),
                round(throttle, 4),
                round(brake, 4),
                round(driven_distance, 2),
                round(route_progress, 2),
                round(yolo_ms, 2),
                yolo_age_frames,
                0 if yolo_boxes is None else len(yolo_boxes),
                "" if lead is None else round(lead[0], 2),
                current_collisions,
                current_invasions,
            ])
            if frames % 20 == 0:
                telemetry.flush()

            display = frame.copy()
            if yolo_boxes is not None and yolo_classes is not None:
                class_names = {
                    0: "person", 1: "bicycle", 2: "car",
                    3: "motorcycle", 5: "bus", 7: "truck",
                }
                for box, class_id in zip(yolo_boxes, yolo_classes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 1)
                    label = class_names.get(int(class_id), str(int(class_id)))
                    cv2.putText(
                        display, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1,
                    )
            if lead is not None:
                cv2.putText(
                    display, f"Lead {lead[0]:.1f} m", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                )
            cv2.putText(
                display,
                f"{mode['name']} | {speed:.1f} km/h | model {model_steer:+.2f} "
                f"| steer {command_steer:+.2f}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                display,
                f"Route {route_progress / 1000:.2f}/{planned_distance / 1000:.2f} km | "
                f"route {route_steer:+.2f} | YOLO age {yolo_age_frames}f",
                (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            if writer is None and args.video:
                height, width = display.shape[:2]
                writer = cv2.VideoWriter(
                    args.video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1.0 / args.fixed_delta,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video output: {args.video}")
            if writer is not None:
                writer.write(display)
            cv2.imshow("CARLA fixed-route PilotNet test", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r") and mode["name"] != "route":
                mode["name"] = "route"
                print("[mode] Route-controller steering active")
            elif key == ord("m") and mode["name"] != "model":
                mode["name"] = "model"
                print("[mode] PilotNet steering active; route completion is not guaranteed")

            if agent.done():
                print("[finish] Fixed 5 km route reached")
                break
            if args.max_seconds and sim_time >= args.max_seconds:
                print("[exit] Reached --max-seconds")
                break

    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[exit] Interrupted by user")
    except Exception as exc:
        status = "error"
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"\n[error] {error_text}")
        traceback.print_exc()
    finally:
        if yolo is not None:
            yolo.stop()
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception as exc:
                print(f"[cleanup] Could not stop sensor: {exc}")
        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                print(f"[cleanup] Could not release video writer: {exc}")
        if telemetry is not None:
            try:
                telemetry.flush()
                telemetry.close()
            except Exception as exc:
                print(f"[cleanup] Could not close route CSV: {exc}")
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                print(f"[cleanup] Could not close OpenCV window: {exc}")
        if client is not None:
            safe_destroy(client, carla, sensors + actors)
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                print(f"[cleanup] Could not disable Traffic Manager sync: {exc}")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as exc:
                print(f"[cleanup] Could not restore CARLA settings: {exc}")

        csv_summary = csv_path if telemetry is not None else "not created"
        print(
            f"[summary] status={status} route={ROUTE_TOWN} "
            f"planned_m={planned_distance:.1f} "
            f"driven_m={driven_distance:.1f} model_mode_s={model_time:.1f} "
            f"frames={frames} csv={csv_summary}"
        )
        if error_text:
            print(f"[summary] error={error_text}")

    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
