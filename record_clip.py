import carla, random, os, queue, time

SECONDS, FPS = 30, 20
WEATHER = carla.WeatherParameters.ClearNoon   # try HardRainNoon or ClearNight
NUM_VEHICLES, NUM_WALKERS = 30, 20
OUT = "clip_frames"

os.makedirs(OUT, exist_ok=True)
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()
bp = world.get_blueprint_library()
spawns = world.get_map().get_spawn_points()
random.shuffle(spawns)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / FPS
world.apply_settings(settings)
tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)
world.set_weather(WEATHER)

actors, walkers, controllers, cam = [], [], [], None
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

    for _ in range(NUM_WALKERS):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(random.choice(bp.filter("walker.pedestrian.*")), carla.Transform(loc))
        if w:
            walkers.append(w)
    world.tick()
    for w in walkers:
        controllers.append(world.spawn_actor(bp.find("controller.ai.walker"), carla.Transform(), attach_to=w))
    world.tick()
    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.2 + random.random() * 0.8)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.0)), attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)

    for _ in range(60):          # 3 s warm-up so the car is moving and traffic is spread out
        world.tick(); q.get(timeout=10)

    for i in range(SECONDS * FPS):
        world.tick()
        q.get(timeout=10).save_to_disk(f"{OUT}/{i:05d}.png")
    print("recorded", SECONDS * FPS, "frames")
finally:
    if cam:
        cam.stop()
    time.sleep(2)
    for c in controllers:
        c.stop()
    client.apply_batch([carla.command.DestroyActor(x.id) for x in controllers + walkers + actors])
    if cam:
        cam.destroy()
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False
    world.apply_settings(settings)