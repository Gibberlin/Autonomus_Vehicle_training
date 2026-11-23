"""Synchronous CARLA capture for aligned multimodal control training samples."""

import argparse
import json
import math
import queue
import random
import time
from pathlib import Path

import cv2
import numpy as np


def receive_frame(sensor_queue, frame_number, timeout):
    while True:
        try:
            data = sensor_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Sensor did not produce world frame {frame_number}"
            ) from exc
        if data.frame == frame_number:
            return data
        if data.frame > frame_number:
            raise RuntimeError(
                f"Sensor frame mismatch: expected {frame_number}, got {data.frame}"
            )


def decode_depth(raw, height, width):
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    blue, green, red = bgra[..., 0], bgra[..., 1], bgra[..., 2]
    return (
        (red.astype(np.float32) + green.astype(np.float32) * 256.0
         + blue.astype(np.float32) * 65536.0)
        / 16777215.0
        * 1000.0
    ).astype(np.float32)


def decode_semantic(raw, height, width):
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return bgra[..., 2].copy()


def radar_points(measurement):
    rows = []
    for detection in measurement:
        cos_alt = math.cos(detection.altitude)
        rows.append((
            detection.depth * cos_alt * math.cos(detection.azimuth),
            detection.depth * cos_alt * math.sin(detection.azimuth),
            detection.depth * math.sin(detection.altitude),
            detection.velocity,
        ))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def make_sensor(blueprints, world, carla, sensor_id, attach_to, transform, **attrs):
    blueprint = blueprints.find(sensor_id)
    for key, value in attrs.items():
        blueprint.set_attribute(key, str(value))
    return world.spawn_actor(blueprint, transform, attach_to=attach_to)


def state_for(ego):
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    forward_speed = (
        forward.x * velocity.x + forward.y * velocity.y + forward.z * velocity.z
    )
    yaw_rate = math.radians(ego.get_angular_velocity().z)
    return {
        "speed_kmh": 3.6 * math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        ),
        "forward_speed_mps": forward_speed,
        "heading_deg": transform.rotation.yaw % 360.0,
        "yaw_rate_rad_s": yaw_rate,
        "forward_acceleration_mps2": (
            forward.x * acceleration.x + forward.y * acceleration.y
            + forward.z * acceleration.z
        ),
        "lateral_acceleration_mps2": (
            right.x * acceleration.x + right.y * acceleration.y
            + right.z * acceleration.z
        ),
        "traffic_light_state": int(ego.get_traffic_light_state()),
        "speed_limit_kmh": float(ego.get_speed_limit()),
    }


def random_weather(carla, rng):
    weather = carla.WeatherParameters()
    weather.cloudiness = rng.uniform(0, 85)
    weather.precipitation = rng.choice((0.0, 0.0, 0.0, rng.uniform(5, 50)))
    weather.precipitation_deposits = rng.uniform(0, 35)
    weather.wetness = max(weather.precipitation_deposits, weather.precipitation)
    weather.wind_intensity = rng.uniform(0, 60)
    weather.sun_altitude_angle = rng.uniform(10, 85)
    weather.sun_azimuth_angle = float(rng.randrange(360))
    return weather


def capture_episode(client, world, carla, args, episode_id, manifest, rng):
    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    if len(spawns) < 2:
        raise RuntimeError("The selected town needs at least two vehicle spawn points")
    rng.shuffle(spawns)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    original_settings = world.get_settings()
    original_weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.fixed_delta
    world.apply_settings(settings)

    actors = []
    queues = {}
    sensor_names = ("rgb", "depth", "semantic", "lidar", "radar")
    episode_dir = Path(args.out) / episode_id

    try:
        world.set_weather(random_weather(carla, rng))
        for name in ("rgb", "depth", "semantic", "lidar", "radar"):
            (episode_dir / name).mkdir(parents=True, exist_ok=True)
        ego = world.try_spawn_actor(bp.find("vehicle.tesla.model3"), spawns.pop())
        if ego is None:
            raise RuntimeError(f"Could not spawn ego vehicle for {episode_id}")
        actors.append(ego)
        ego.set_autopilot(True, tm.get_port())

        vehicle_bps = [
            item for item in bp.filter("vehicle.*")
            if item.has_attribute("number_of_wheels")
            and int(item.get_attribute("number_of_wheels")) == 4
        ]
        episode_vehicle_count = rng.randint(args.vehicles // 2, args.vehicles)
        spawned = 0
        for spawn in spawns[:episode_vehicle_count]:
            npc = world.try_spawn_actor(rng.choice(vehicle_bps), spawn)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                actors.append(npc)
                spawned += 1
        walker_controller_bp = bp.find("controller.ai.walker")
        walker_blueprints = bp.filter("walker.pedestrian.*")
        walker_count = rng.randint(args.walkers // 2, args.walkers)
        spawned_walkers = 0
        walker_controllers = []
        for _ in range(walker_count):
            location = world.get_random_location_from_navigation()
            if location is None:
                continue
            walker = world.try_spawn_actor(
                rng.choice(walker_blueprints), carla.Transform(location)
            )
            if walker is None:
                continue
            actors.append(walker)
            controller = world.spawn_actor(
                walker_controller_bp, carla.Transform(), attach_to=walker
            )
            actors.append(controller)
            walker_controllers.append(controller)
            spawned_walkers += 1
        if walker_controllers:
            world.tick()
        for controller in walker_controllers:
            destination = world.get_random_location_from_navigation()
            controller.start()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(rng.uniform(1.0, 1.8))

        carla_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        common = {"image_size_x": args.width, "image_size_y": args.height,
                  "fov": args.fov, "sensor_tick": args.fixed_delta}
        sensors = {}
        specifications = (
            ("rgb", "sensor.camera.rgb", carla_transform, common),
            ("depth", "sensor.camera.depth", carla_transform, common),
            ("semantic", "sensor.camera.semantic_segmentation", carla_transform, common),
            ("lidar", "sensor.lidar.ray_cast",
             carla.Transform(carla.Location(x=0.0, z=2.2)), {
                 "channels": args.lidar_channels,
                 "points_per_second": args.lidar_points_per_second,
                 "rotation_frequency": 1.0 / args.fixed_delta,
                 "range": args.lidar_range,
                 "sensor_tick": args.fixed_delta,
             }),
            ("radar", "sensor.other.radar", carla_transform, {
                "horizontal_fov": args.radar_fov,
                "vertical_fov": 20.0,
                "range": args.radar_range,
                "sensor_tick": args.fixed_delta,
            }),
        )
        for name, sensor_id, sensor_transform, attributes in specifications:
            sensors[name] = make_sensor(
                bp, world, carla, sensor_id, ego, sensor_transform, **attributes
            )
            actors.append(sensors[name])
        for name, sensor in sensors.items():
            queues[name] = queue.Queue()
            sensor.listen(queues[name].put)

        for _ in range(args.warmup):
            frame = world.tick()
            for name in sensor_names:
                receive_frame(queues[name], frame, args.sensor_timeout)

        samples = 0
        next_recovery = args.recovery_interval or args.samples_per_episode + 1
        recovery_end = -1
        intervention_left = 0
        intervention_steer = 0.0
        frame_started = time.monotonic()
        while samples < args.samples_per_episode:
            if intervention_left == 0 and samples >= next_recovery:
                intervention_left = args.recovery_ticks
                intervention_steer = rng.choice((-1.0, 1.0)) * rng.uniform(
                    args.recovery_steer_min, args.recovery_steer_max
                )
                ego.set_autopilot(False)
            if intervention_left > 0:
                ego.apply_control(carla.VehicleControl(
                    steer=intervention_steer,
                    throttle=0.15,
                    brake=0.0,
                ))

            frame = world.tick()
            sensor_data = {
                name: receive_frame(queues[name], frame, args.sensor_timeout)
                for name in sensor_names
            }
            if intervention_left > 0:
                intervention_left -= 1
                if intervention_left == 0:
                    ego.apply_control(carla.VehicleControl())
                    ego.set_autopilot(True, tm.get_port())
                    recovery_end = samples + args.recovery_label_frames
                    next_recovery = samples + args.recovery_interval
                continue

            stem = f"{samples:06d}"
            rgb_image = sensor_data["rgb"]
            bgra = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape(
                args.height, args.width, 4
            )
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            rgb_path = episode_dir / "rgb" / f"{stem}.jpg"
            if not cv2.imwrite(
                str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise OSError(f"Could not save frame: {rgb_path}")

            depth_path = episode_dir / "depth" / f"{stem}.npy"
            np.save(depth_path, decode_depth(
                sensor_data["depth"].raw_data, args.height, args.width
            ))
            semantic_path = episode_dir / "semantic" / f"{stem}.npy"
            np.save(semantic_path, decode_semantic(
                sensor_data["semantic"].raw_data, args.height, args.width
            ))
            lidar_path = episode_dir / "lidar" / f"{stem}.npy"
            lidar_data = np.frombuffer(
                sensor_data["lidar"].raw_data, dtype=np.float32
            ).reshape(-1, 4)
            np.save(lidar_path, lidar_data)
            radar_path = episode_dir / "radar" / f"{stem}.npy"
            np.save(radar_path, radar_points(sensor_data["radar"]))

            control = ego.get_control()
            row = {
                "episode_id": episode_id,
                "frame": int(frame),
                "timestamp_s": round(frame * args.fixed_delta, 6),
                "weather": {
                    "cloudiness": world.get_weather().cloudiness,
                    "precipitation": world.get_weather().precipitation,
                    "wetness": world.get_weather().wetness,
                    "sun_altitude_angle": world.get_weather().sun_altitude_angle,
                    "sun_azimuth_angle": world.get_weather().sun_azimuth_angle,
                },
                "recovery": int(samples <= recovery_end),
                "rgb": rgb_path.relative_to(args.out).as_posix(),
                "depth": depth_path.relative_to(args.out).as_posix(),
                "semantic": semantic_path.relative_to(args.out).as_posix(),
                "lidar": lidar_path.relative_to(args.out).as_posix(),
                "radar": radar_path.relative_to(args.out).as_posix(),
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                **state_for(ego),
            }
            manifest.write(json.dumps(row) + "\n")
            samples += 1
            if samples % 100 == 0:
                elapsed = time.monotonic() - frame_started
                print(
                    f"[{episode_id}] {samples}/{args.samples_per_episode} "
                    f"samples; NPCs={spawned}, walkers={spawned_walkers}; "
                    f"{samples / max(elapsed, 1e-6):.1f} Hz"
                )

    finally:
        for actor in actors:
            if actor.is_alive and hasattr(actor, "stop"):
                actor.stop()
        alive = [actor for actor in actors if actor.is_alive]
        if alive:
            client.apply_batch([
                carla.command.DestroyActor(actor.id) for actor in alive
            ])
        tm.set_synchronous_mode(False)
        world.set_weather(original_weather)
        world.apply_settings(original_settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--towns", default="Town03",
                        help="comma-separated CARLA towns to collect from")
    parser.add_argument("--out", type=Path, default=Path("fusion_data"))
    parser.add_argument("--episodes-per-town", type=int, default=3)
    parser.add_argument("--samples-per-episode", type=int, default=1200)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fov", type=float, default=90)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--sensor-timeout", type=float, default=15)
    parser.add_argument("--lidar-channels", type=int, default=32)
    parser.add_argument("--lidar-points-per-second", type=int, default=56000)
    parser.add_argument("--lidar-range", type=float, default=50)
    parser.add_argument("--radar-fov", type=float, default=35)
    parser.add_argument("--radar-range", type=float, default=80)
    parser.add_argument("--recovery-interval", type=int, default=300,
                        help="inject a brief steering disturbance every N samples; 0 disables")
    parser.add_argument("--recovery-ticks", type=int, default=3)
    parser.add_argument("--recovery-label-frames", type=int, default=40)
    parser.add_argument("--recovery-steer-min", type=float, default=0.15)
    parser.add_argument("--recovery-steer-max", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.fixed_delta <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fixed delta and image dimensions must be positive")
    if args.recovery_interval < 0 or args.recovery_ticks <= 0:
        parser.error("recovery interval must be non-negative and ticks positive")
    if (args.episodes_per_town <= 0 or args.samples_per_episode <= 0
            or args.vehicles < 0 or args.walkers < 0):
        parser.error("episode/sample counts must be positive and traffic counts non-negative")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    towns = [town.strip() for town in args.towns.split(",") if town.strip()]
    if not towns:
        parser.error("--towns must contain at least one town")
    rng = random.Random(args.seed)

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API unavailable; run with CARLA's Python environment"
        ) from exc

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    original_world = client.get_world()
    original_town = original_world.get_map().name.rsplit("/", 1)[-1]
    original_settings = original_world.get_settings()
    original_weather = original_world.get_weather()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for town in towns:
                world = client.load_world(town)
                for episode in range(args.episodes_per_town):
                    episode_id = f"{town}_{run_id}_episode_{episode:03d}"
                    print(f"[capture] {episode_id}")
                    capture_episode(
                        client, world, carla, args, episode_id, manifest, rng
                    )
                    manifest.flush()
    finally:
        current_world = client.get_world()
        if current_world.get_map().name.rsplit("/", 1)[-1] != original_town:
            current_world = client.load_world(original_town)
        current_world.set_weather(original_weather)
        current_world.apply_settings(original_settings)
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
