import carla, random, os, csv, queue, time
import numpy as np, cv2

TOWN = "Town03"          # set to None to keep the currently loaded map
SAMPLES = 3000           # saved samples (x3 images each); ~10 Hz, so ~5 min of driving
NUM_VEHICLES = 25
W, H, SAVE_EVERY = 320, 240, 2     # 20 Hz sim, save every 2nd tick = 10 Hz
OUT = "drive_data"

os.makedirs(f"{OUT}/img", exist_ok=True)
client = carla.Client("localhost", 2000)
client.set_timeout(60.0)
world = client.load_world(TOWN) if TOWN else client.get_world()
bp = world.get_blueprint_library()
spawns = world.get_map().get_spawn_points()
random.shuffle(spawns)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)
world.set_weather(carla.WeatherParameters.ClearNoon)

actors, cams = [], []
try:
    ego = world.spawn_actor(bp.find("vehicle.tesla.model3"), spawns.pop())
    ego.set_autopilot(True, tm.get_port())
    actors.append(ego)
    cars = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    for sp in spawns[:NUM_VEHICLES]:
        v = world.try_spawn_actor(random.choice(cars), sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(W))
    cam_bp.set_attribute("image_size_y", str(H))
    cam_bp.set_attribute("fov", "90")
    queues = {}
    for name, y in {"center": 0.0, "left": -0.5, "right": 0.5}.items():
        c = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, y=y, z=2.0)), attach_to=ego)
        queues[name] = queue.Queue()
        c.listen(queues[name].put)
        cams.append(c)

    for _ in range(60):
        world.tick()
        for q in queues.values(): q.get(timeout=10)

    saved, tick = 0, 0
    with open(f"{OUT}/labels.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "cam", "file", "steer", "throttle", "brake", "speed"])
        while saved < SAMPLES:
            world.tick(); tick += 1
            imgs = {n: q.get(timeout=10) for n, q in queues.items()}
            if tick % SAVE_EVERY:
                continue
            ctrl, vel = ego.get_control(), ego.get_velocity()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            if speed < 0.5 and random.random() > 0.1:      # keep only 10% of stopped frames
                continue
            for n, im in imgs.items():
                arr = np.frombuffer(im.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3]
                fn = f"{im.frame:07d}_{n}.jpg"
                cv2.imwrite(f"{OUT}/img/{fn}", arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                wr.writerow([im.frame, n, fn, round(ctrl.steer, 4), round(ctrl.throttle, 3),
                             round(ctrl.brake, 3), round(speed, 2)])
            saved += 1
            if saved % 500 == 0:
                print(saved, "samples")
finally:
    for c in cams: c.stop()
    time.sleep(2)
    client.apply_batch([carla.command.DestroyActor(x.id) for x in cams + actors])
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False
    world.apply_settings(settings)