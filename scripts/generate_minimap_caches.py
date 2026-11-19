#!/usr/bin/env python3
"""Generate map-specific CARLA road caches and minimap preview images."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import carla
import cv2

from visualization.minimap import (
    Minimap,
    map_cache_key,
    road_cache_path_for_map,
)


def map_short_name(map_name):
    return map_name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate minimap caches for all drivable maps available to CARLA"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--size", type=int, default=800, help="preview image size in pixels")
    parser.add_argument("--preview-dir", default=os.path.join("results", "minimap_previews"))
    parser.add_argument("--maps", nargs="*", help="optional subset of CARLA map names")
    parser.add_argument("--force", action="store_true", help="regenerate existing road caches")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    original_map_name = map_short_name(client.get_world().get_map().name)

    available_maps = {
        map_short_name(map_name): map_name
        for map_name in client.get_available_maps()
        if not map_short_name(map_name).casefold().startswith("annotation")
    }
    if args.maps:
        available_by_key = {
            map_cache_key(name): name for name in available_maps
        }
        requested_keys = [map_cache_key(name) for name in args.maps]
        missing = [name for name, key in zip(args.maps, requested_keys)
                   if key not in available_by_key]
        if missing:
            raise ValueError(f"Unknown CARLA maps: {', '.join(missing)}")
        map_names = [available_by_key[key] for key in requested_keys]
    else:
        map_names = sorted(available_maps.values(), key=str.casefold)
    if not map_names:
        raise RuntimeError("CARLA reported no drivable maps")

    os.makedirs(args.preview_dir, exist_ok=True)
    print(f"[maps] Generating {len(map_names)} map minimaps: {', '.join(map_names)}")

    try:
        for map_name in map_names:
            cache_path = road_cache_path_for_map(map_name)
            map_obj = None
            if args.force or not os.path.isfile(cache_path):
                world = client.load_world(map_name)
                map_obj = world.get_map()

            minimap = Minimap(
                size=args.size,
                carla_map=map_obj,
                road_cache_path=cache_path,
                map_name=map_name,
            )
            if minimap.base_image is None:
                raise RuntimeError(
                    f"Could not render minimap for {map_name}; OpenCV is unavailable"
                )

            preview_path = os.path.join(
                args.preview_dir, f"{map_cache_key(map_name)}.png"
            )
            if not cv2.imwrite(preview_path, minimap.base_image):
                raise RuntimeError(f"Could not save minimap preview: {preview_path}")
            print(
                f"[map] {map_name}: {len(minimap.roads)} road polylines, "
                f"Start={minimap.start_location}, "
                f"Destination={minimap.destination_location}, "
                f"cache={cache_path}, preview={preview_path}"
            )
    finally:
        current_map_name = map_short_name(client.get_world().get_map().name)
        if current_map_name.casefold() != original_map_name.casefold():
            print(f"[restore] Returning CARLA server to {original_map_name}")
            restored_world = client.load_world(original_map_name)
            restored_name = map_short_name(restored_world.get_map().name)
            if restored_name.casefold() != original_map_name.casefold():
                raise RuntimeError(
                    f"CARLA restored {restored_name}, expected {original_map_name}"
                )

    print("[done] All requested map caches and previews generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
