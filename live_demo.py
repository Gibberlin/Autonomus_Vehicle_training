import argparse
import csv
import json
import os
import queue
import random
import threading
import time
import traceback
from datetime import datetime

import numpy as np


CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_FOV = 90.0
CAMERA_HEIGHT_M = 2.0
LANE_HALF_WIDTH_M = 1.8
DETECTION_CLASSES = [0, 1, 2, 3, 5, 7]
LEAD_CLASSES = {2, 3, 5, 7}


class YoloWorker:
    def __init__(self, weights, device, half):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.half = half
        self.input = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="yolo-inference", daemon=True)
        self.thread.start()

    def submit(self, frame_number, frame):
        try:
            self.input.put_nowait((frame_number, frame.copy()))
        except queue.Full:
            try:
                self.input.get_nowait()
            except queue.Empty:
                pass
            self.input.put_nowait((frame_number, frame.copy()))

    def latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("YOLO inference failed") from self.error
            return self.result

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame_number, frame = self.input.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                inference_args = {
                    "device": self.device,
                    "verbose": False,
                    "classes": DETECTION_CLASSES,
                    "imgsz": 320,
                }
                if self.half:
                    inference_args["quantize"] = 16
                result = self.model(
                    frame,
                    **inference_args,
                )[0]
                elapsed_ms = (time.perf_counter() - started) * 1000
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                item = (frame_number, elapsed_ms, boxes, classes)
                with self.lock:
                    self.result = item
            except Exception as exc:
                with self.lock:
                    self.error = exc
                self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            print("[cleanup] YOLO worker did not stop within 5 seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Live CARLA PilotNet / autopilot demo")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default=None, help="optionally load a map such as Town03")
    parser.add_argument("--weights", default="steer_model.pt", help="PilotNet state-dict path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--no-yolo", action="store_true", help="disable object detection")
    parser.add_argument("--vehicles", type=int, default=20, help="number of NPC vehicles")
    parser.add_argument("--target-speed", type=float, default=25.0, help="model-mode speed in km/h")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="stop after this simulation time")
    parser.add_argument("--fixed-delta", type=float, default=0.05, help="simulation step in seconds")
    parser.add_argument("--video", default=None, help="optional output video path")
    parser.add_argument("--no-display", action="store_true", help="run without an OpenCV window")
    parser.add_argument("--start-model", dest="start_autopilot", action="store_false",
                        help="start with PilotNet instead of CARLA autopilot")
    parser.set_defaults(start_autopilot=True)
    parser.add_argument("--minimap", action="store_true", help="enable Town03 road minimap overlay")
    parser.add_argument("--minimap-position", choices=("top-right", "top-left"), default="top-right",
                        help="minimap overlay position on display (default: top-right)")
    parser.add_argument("--minimap-size", type=int, default=260,
                        help="minimap pixel dimensions (width and height, default: 260)")
    parser.add_argument("--minimap-opacity", type=float, default=0.85,
                        help="minimap overlay opacity between 0.0 and 1.0 (default: 0.85)")
    parser.add_argument("--checkpoints-config", default=None,
                        help="optional path to custom checkpoints JSON file")
    parser.add_argument("--refresh-map-cache", action="store_true",
                        help="force re-extracting map road topology from live CARLA world")
    return parser.parse_args()


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


def find_lead(boxes, classes, width, height):
    cx, cy = width / 2.0, height / 2.0
    focal = width / (2.0 * np.tan(np.radians(CAMERA_FOV / 2.0)))
    lead = None
    for box, class_id in zip(boxes, classes):
        x1, _, x2, y2 = box
        if class_id not in LEAD_CLASSES or y2 <= cy + 3:
            continue
        distance = focal * CAMERA_HEIGHT_M / (y2 - cy)
        lateral = ((x1 + x2) / 2.0 - cx) * distance / focal
        if abs(lateral) < LANE_HALF_WIDTH_M and (lead is None or distance < lead[0]):
            lead = (distance, x1, y2, x2)
    return lead


def draw_distance_guide(frame, cv2, lead_distance=None):
    height, width = frame.shape[:2]
    center_x, horizon_y = width / 2.0, height / 2.0
    focal = width / (2.0 * np.tan(np.radians(CAMERA_FOV / 2.0)))

    def ground(distance, lateral):
        x = center_x + focal * lateral / distance
        y = horizon_y + focal * CAMERA_HEIGHT_M / distance
        return int(round(x)), int(round(y))

    def lead_color(distance):
        if distance < 8:
            return (0, 0, 255)
        if distance < 15:
            return (0, 255, 255)
        return (0, 255, 0)

    far = min(max(3.0, lead_distance), 30.0) if lead_distance is not None else 30.0
    shade_color = lead_color(lead_distance) if lead_distance is not None else (255, 200, 0)
    polygon = np.array([
        ground(3.0, -LANE_HALF_WIDTH_M),
        ground(3.0, LANE_HALF_WIDTH_M),
        ground(far, LANE_HALF_WIDTH_M),
        ground(far, -LANE_HALF_WIDTH_M),
    ], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], shade_color)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0.0, frame)

    for distance in (5, 10, 20, 30):
        left = ground(distance, -LANE_HALF_WIDTH_M)
        right = ground(distance, LANE_HALF_WIDTH_M)
        cv2.line(frame, left, right, (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"{distance}m",
            (left[0] - 38, left[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    return frame


def main():
    args = parse_args()

    try:
        import cv2
        import torch
    except ImportError as exc:
        raise RuntimeError("This demo requires OpenCV and PyTorch to be installed") from exc

    try:
        from pilotnet import PilotNet, preprocess
    except ImportError as exc:
        raise RuntimeError("pilotnet.py must be available in the same directory as live_demo.py") from exc

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("CARLA Python API is unavailable; run with CARLA's Python environment") from exc

    if args.fixed_delta <= 0:
        raise ValueError("--fixed-delta must be greater than zero")
    if args.vehicles < 0:
        raise ValueError("--vehicles cannot be negative")
    if args.target_speed < 0:
        raise ValueError("--target-speed cannot be negative")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"PilotNet weights not found: {args.weights}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = PilotNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    yolo_device = args.yolo_device or ("0" if device == "cuda" else "cpu")
    yolo = None if args.no_yolo else YoloWorker(args.yolo_weights, yolo_device, args.yolo_half)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"live_{stamp}.csv")

    client = world = tm = ego = None
    original_settings = None
    sensors, actors = [], []
    camera_queue = queue.Queue(maxsize=1)
    writer = telemetry = None
    status, error_text = "ok", ""
    sim_time_run = model_time_run = distance = model_distance_run = 0.0
    frames_run = yolo_runs_run = 0
    pipeline_sum_run = yolo_sum_run = 0.0
    previous_location = None
    steer_previous = 0.0
    state = {"auto": args.start_autopilot}
    stats = {"collisions": 0, "invasions": 0, "collisions_model": 0, "invasions_model": 0}
    stats_lock = threading.Lock()
    started_at = time.perf_counter()
    yolo_frame_number = -1
    yolo_ms = 0.0
    yolo_boxes = np.empty((0, 4), dtype=np.float32)
    yolo_classes = np.empty((0,), dtype=int)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.load_world(args.town) if args.town else client.get_world()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)

        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        blueprints = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("The current CARLA map has no vehicle spawn points")
        random.shuffle(spawn_points)

        ego_bp = blueprints.find("vehicle.tesla.model3")
        ego = world.try_spawn_actor(ego_bp, spawn_points.pop())
        if ego is None:
            raise RuntimeError("Could not spawn the ego vehicle at the selected spawn point")
        actors.append(ego)
        ego.set_autopilot(state["auto"], tm.get_port())

        vehicle_bps = [
            blueprint for blueprint in blueprints.filter("vehicle.*")
            if int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        for spawn in spawn_points[:args.vehicles]:
            npc = world.try_spawn_actor(random.choice(vehicle_bps), spawn)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                actors.append(npc)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=CAMERA_HEIGHT_M)),
            attach_to=ego,
        )
        sensors.append(camera)
        camera.listen(camera_queue.put)

        collision_bp = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        sensors.append(collision_sensor)

        lane_bp = blueprints.find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)
        sensors.append(lane_sensor)

        def record_event(_event, stat_key, model_key):
            _ = _event
            with stats_lock:
                stats[stat_key] += 1
                if not state["auto"]:
                    stats[model_key] += 1

        collision_sensor.listen(
            lambda event: record_event(event, "collisions", "collisions_model")
        )
        lane_sensor.listen(
            lambda event: record_event(event, "invasions", "invasions_model")
        )

        telemetry = open(csv_path, "w", newline="", encoding="utf-8")
        telemetry_writer = csv.writer(telemetry)
        telemetry_writer.writerow([
            "sim_time_s", "world_frame", "camera_frame", "sensor_lag_frames",
            "mode", "speed_kmh", "autopilot_steer", "steer_raw",
            "steer_command", "throttle", "brake", "steer_ms", "yolo_ms",
            "yolo_age_frames", "pipeline_ms", "detections", "lead_distance_m",
            "collisions", "lane_invasions", "distance_m",
        ])

        minimap = None
        if args.minimap:
            try:
                from visualization.minimap import Minimap
                from navigation.checkpoints import load_or_generate_checkpoints

                cp_config = args.checkpoints_config or os.path.join("config", "checkpoints_town03.json")
                checkpoints = load_or_generate_checkpoints(world.get_map(), cp_config)
                carla_map_arg = world.get_map() if args.refresh_map_cache else None
                minimap = Minimap(
                    size=args.minimap_size,
                    position=args.minimap_position,
                    opacity=args.minimap_opacity,
                    carla_map=carla_map_arg,
                    checkpoints=checkpoints,
                    checkpoints_path=cp_config,
                )
                print(
                    f"[minimap] Enabled ({args.minimap_position}, size={args.minimap_size}px, "
                    f"opacity={args.minimap_opacity}, {len(checkpoints)} checkpoints)"
                )
            except Exception as exc:
                print(f"[minimap] Warning: Could not initialize minimap: {exc}")
                traceback.print_exc()
                minimap = None

        print("[ready] CARLA live demo")
        print("[keys] m: PilotNet | a: autopilot | q / Esc: quit")
        while True:
            world_frame = world.tick()
            image = camera_queue.get(timeout=10.0)
            camera_frame = int(image.frame)
            sensor_lag = max(0, int(world_frame) - camera_frame)
            frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3]
            frame = np.ascontiguousarray(frame)
            if writer is None and args.video:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / args.fixed_delta,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video output: {args.video}")

            pipeline_start = time.perf_counter()
            if yolo is not None:
                yolo.submit(camera_frame, frame)
                latest = yolo.latest()
                if latest is not None:
                    yolo_frame_number, yolo_ms, yolo_boxes, yolo_classes = latest
            else:
                yolo_frame_number, yolo_ms = camera_frame, 0.0
                yolo_boxes = np.empty((0, 4), dtype=np.float32)
                yolo_classes = np.empty((0,), dtype=int)

            lead = find_lead(yolo_boxes, yolo_classes, frame.shape[1], frame.shape[0])
            speed = speed_kmh(ego)
            location = ego.get_location()
            if previous_location is not None:
                step_distance = location.distance(previous_location)
                distance += step_distance
                if not state["auto"]:
                    model_distance_run += step_distance
            previous_location = location

            model_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            steer_input = preprocess(model_frame)
            steer_tensor = torch.from_numpy(steer_input).unsqueeze(0).to(device).float() / 255.0
            steer_start = time.perf_counter()
            with torch.no_grad():
                steer_raw = float(model(steer_tensor).item())
            if device == "cuda":
                torch.cuda.synchronize()
            steer_ms = (time.perf_counter() - steer_start) * 1000
            steer_command = float(np.clip(0.75 * steer_previous + 0.25 * steer_raw, -1.0, 1.0))
            steer_previous = steer_command

            if state["auto"]:
                control = ego.get_control()
                steer_autopilot = float(control.steer)
                throttle, brake = float(control.throttle), float(control.brake)
            else:
                steer_autopilot = None
                speed_error = args.target_speed - speed
                throttle = float(np.clip(speed_error * 0.04, 0.0, 0.5))
                brake = float(np.clip(-speed_error * 0.04, 0.0, 1.0))
                ego.apply_control(carla.VehicleControl(
                    throttle=throttle,
                    steer=steer_command,
                    brake=brake,
                ))
                model_time_run += args.fixed_delta

            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
            sim_time_run += args.fixed_delta
            frames_run += 1
            pipeline_sum_run += pipeline_ms
            if yolo is not None and yolo_frame_number >= 0:
                yolo_runs_run += 1
                yolo_sum_run += yolo_ms
            yolo_age_frames = max(0, camera_frame - yolo_frame_number) if yolo_frame_number >= 0 else 0

            with stats_lock:
                collision_count = stats["collisions"]
                invasion_count = stats["invasions"]
            mode_name = "autopilot" if state["auto"] else "model"
            telemetry_writer.writerow([
                round(sim_time_run, 2),
                int(world_frame),
                int(camera_frame),
                int(sensor_lag),
                mode_name,
                round(speed, 3),
                "" if steer_autopilot is None else round(steer_autopilot, 4),
                round(steer_raw, 4),
                round(steer_command, 4),
                round(throttle, 4),
                round(brake, 4),
                round(steer_ms, 2),
                round(yolo_ms, 2),
                int(yolo_age_frames),
                round(pipeline_ms, 2),
                len(yolo_boxes),
                "" if lead is None else round(lead[0], 2),
                collision_count,
                invasion_count,
                round(distance, 2),
            ])
            if frames_run % 20 == 0:
                telemetry.flush()

            display = frame.copy()
            draw_distance_guide(
                display, cv2, None if lead is None else lead[0]
            )
            for box, class_id in zip(yolo_boxes, yolo_classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 1)
                cv2.putText(display, str(class_id), (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
            if lead is not None:
                lead_distance, x1, y2, x2 = lead
                cv2.line(display, (int(x1), int(y2)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(display, f"{lead_distance:.1f} m",
                            (int(x1), min(int(y2) + 18, display.shape[0] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(
                display,
                f"{mode_name} | {speed:.1f} km/h | steer {steer_command:+.2f} | "
                f"YOLO age {yolo_age_frames}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            if minimap is not None:
                ego_tf = ego.get_transform()
                display = minimap.overlay_on_frame(
                    display,
                    vehicle_x=ego_tf.location.x,
                    vehicle_y=ego_tf.location.y,
                    vehicle_yaw=ego_tf.rotation.yaw,
                )
            if writer is not None:
                writer.write(display)
            if not args.no_display:
                cv2.imshow("CARLA live demo", display)

            key = cv2.waitKey(1) & 0xFF if not args.no_display else 255
            if key in (27, ord("q")) or (args.max_seconds and sim_time_run >= args.max_seconds):
                break
            if key == ord("m") and state["auto"]:
                ego.set_autopilot(False)
                ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
                state["auto"] = False
                steer_previous = 0.0
                print("[mode] PilotNet now has control")
            elif key == ord("a") and not state["auto"]:
                ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
                ego.set_autopilot(True, tm.get_port())
                state["auto"] = True
                print("[mode] CARLA autopilot now has control")

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
                print(f"[cleanup] Could not close telemetry CSV: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"[cleanup] Could not close OpenCV windows: {exc}")

        if client is not None:
            safe_destroy(client, carla, sensors + actors)
        if tm is not None:
            try:
                tm.set_synchronous_mode(False)
            except Exception as exc:
                print(f"[cleanup] Could not disable Traffic Manager sync: {exc}")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as exc:
                print(f"[cleanup] Could not restore CARLA settings: {exc}")

        with stats_lock:
            final_stats = dict(stats)
        summary = {
            "time": stamp,
            "status": status,
            "error": error_text,
            "weights": args.weights,
            "sim_seconds": round(sim_time_run, 2),
            "dist_m": round(distance, 2),
            "model_driving_s": round(model_time_run, 2),
            "model_driving_m": round(model_distance_run, 2),
            "collisions": final_stats["collisions"],
            "lane_invasions": final_stats["invasions"],
            "collisions_while_model_drove": final_stats["collisions_model"],
            "invasions_while_model_drove": final_stats["invasions_model"],
            "mean_pipeline_ms": round(pipeline_sum_run / max(frames_run, 1), 2),
            "mean_yolo_ms": round(yolo_sum_run / max(yolo_runs_run, 1), 2),
            "frames": frames_run,
            "yolo_runs": yolo_runs_run,
            "csv": csv_path,
            "video": args.video or None,
            "params": vars(args),
            "wall_seconds": round(time.perf_counter() - started_at, 2),
        }
        try:
            with open(os.path.join("results", "live_runs.jsonl"), "a", encoding="utf-8") as summary_file:
                summary_file.write(json.dumps(summary) + "\n")
            print("\n[summary]")
            print(json.dumps(summary, indent=2))
        except Exception as exc:
            print(f"[cleanup] Could not write run summary: {exc}")

    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
