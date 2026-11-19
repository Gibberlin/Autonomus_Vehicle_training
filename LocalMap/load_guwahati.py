import os
import sys

import carla
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualization.minimap import Minimap

# 1. Read your downloaded Guwahati .osm file (must be UTF-8 encoded)
with open(os.path.join(SCRIPT_DIR, "guwahati.osm"), "r", encoding="utf-8") as f:
    osm_data = f.read()

# 2. Configure 0.9.15-compatible conversion settings
settings = carla.Osm2OdrSettings()
settings.default_lane_width = 3.5  # Standard lane width in meters
settings.generate_traffic_lights = True
settings.all_junctions_traffic_lights = False

print("Converting OSM to OpenDRIVE (.xodr)...")
xodr_data = carla.Osm2Odr.convert(osm_data, settings)

# Save the converted .xodr file for future use
with open(os.path.join(SCRIPT_DIR, "Guwahati.xodr"), "w", encoding="utf-8") as f:
    f.write(xodr_data)

# 3. Connect to CARLA 0.9.15 and build the 3D road mesh
print("Connecting to CARLA 0.9.15 and generating roads...")
client = carla.Client("localhost", 2000)
client.set_timeout(120.0)

params = carla.OpendriveGenerationParameters(
    vertex_distance=2.0,
    max_road_length=500.0,
    wall_height=0.0,
    additional_width=0.6,
    smooth_junctions=True,
    enable_mesh_visibility=True
)

world = client.generate_opendrive_world(xodr_data, params)
minimap = Minimap(
    size=800,
    carla_map=world.get_map(),
    map_name=world.get_map().name,
)
if minimap.base_image is None:
    raise RuntimeError("Could not render the Guwahati minimap; OpenCV is unavailable")

preview_path = os.path.join(
    PROJECT_ROOT, "results", "minimap_previews", "guwahati.png"
)
os.makedirs(os.path.dirname(preview_path), exist_ok=True)
if not cv2.imwrite(preview_path, minimap.base_image):
    raise RuntimeError(f"Could not save Guwahati minimap preview: {preview_path}")

print(
    "Done! Guwahati road network loaded. "
    f"Start={minimap.start_location}, Destination={minimap.destination_location}"
)
print(f"Minimap preview saved to {preview_path}")