import carla, random, os, json, queue, time

NUM_VEHICLES, NUM_WALKERS = 40, 30
FRAMES_PER_WEATHER, TICKS_BETWEEN_SAVES = 120, 5
MAX_DIST = 60  # only label actors within 60 m of the ego car

WEATHERS = {
    "clear": carla.WeatherParameters.ClearNoon,
    "rain":  carla.WeatherParameters.HardRainNoon,
    "fog":   carla.WeatherParameters(cloudiness=80, fog_density=60, fog_distance=10, sun_altitude_angle=45),
    "night": carla.WeatherParameters.ClearNight,
}

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/meta", exist_ok=True)

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()
bp = world.get_blueprint_library()
spawns = world.get_map().get_spawn_points()
random.shuffle(spawns)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

actors, walkers, controllers = [], [], []
cam = None
try:
    # Ego vehicle
    ego = world.spawn_actor(bp.find("vehicle.tesla.model3"), spawns.pop())
    ego.set_autopilot(True, tm.get_port())
    actors.append(ego)

    # NPC vehicles
    cars = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    for sp in spawns[:NUM_VEHICLES]:
        v = world.try_spawn_actor(random.choice(cars), sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            actors.append(v)

    # Pedestrians
    ctrl_bp = bp.find("controller.ai.walker")
    for _ in range(NUM_WALKERS):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(random.choice(bp.filter("walker.pedestrian.*")),
                                  carla.Transform(loc))
        if w:
            walkers.append(w)
    world.tick()
    for w in walkers:
        c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        controllers.append(c)
    world.tick()
    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.2 + random.random() * 0.8)

    # Camera
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam_bp.set_attribute("fov", "90")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.0)), attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)

    targets = [a for a in world.get_actors() if a.type_id.startswith(("vehicle.", "walker.pedestrian"))
               and a.id != ego.id]

    for wname, weather in WEATHERS.items():
        world.set_weather(weather)
        for _ in range(40):  # let the scene settle
            world.tick(); q.get(timeout=10)
        for i in range(FRAMES_PER_WEATHER):
            for _ in range(TICKS_BETWEEN_SAVES):
                world.tick(); img = q.get(timeout=10)
            name = f"{wname}_{i:04d}"
            img.save_to_disk(f"dataset/images/{name}.png")

            ego_loc = ego.get_location()
            objs = []
            for a in targets:
                if not a.is_alive or a.get_location().distance(ego_loc) > MAX_DIST:
                    continue
                verts = a.bounding_box.get_world_vertices(a.get_transform())
                objs.append({"id": a.id,
                             "cls": "pedestrian" if a.type_id.startswith("walker") else "vehicle",
                             "verts": [[v.x, v.y, v.z] for v in verts]})
            meta = {"image": f"{name}.png", "width": 640, "height": 480, "fov": 90,
                    "world_to_cam": cam.get_transform().get_inverse_matrix(),
                    "objects": objs}
            with open(f"dataset/meta/{name}.json", "w") as fh:
                json.dump(meta, fh)
        print(f"{wname}: done")
finally:
    if cam:
        cam.stop()
    time.sleep(1)
    for c in controllers:
        c.stop()
    client.apply_batch([carla.command.DestroyActor(x.id) for x in controllers + walkers + actors])
    if cam:
        cam.destroy()
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False
    world.apply_settings(settings)