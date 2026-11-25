#!/usr/bin/env python
"""
Pick CARLA towns from the command line and collect data from each one.

Needs collect_data.py in the same folder, and the CARLA server already running.

Examples:
  python run_collection.py --list                       # show maps on the server
  python run_collection.py                              # interactive menu
  python run_collection.py --towns Town01 Town03 Town10HD --frames 5000
  python run_collection.py --towns all --frames 3000 --out dataset

Output layout:
  dataset/Town01/{rgb,depth,semantic,lidar,radar}/ + labels.csv
  dataset/Town03/...
  dataset/all_labels.csv   (every town's labels, with an extra 'town' column)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECT = os.path.join(HERE, "collect_data.py")
LARGE_MAPS = {"Town12", "Town13", "Town15"}  # need a lot of GPU memory


def get_maps(host, port, include_opt):
    """Retrieve available maps from running CARLA server, with local fallback."""
    maps = []
    import socket
    server_online = False
    try:
        with socket.create_connection((host, port), timeout=1.5):
            server_online = True
    except (OSError, socket.timeout):
        server_online = False

    if server_online:
        try:
            import carla
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            names = client.get_available_maps()
            maps = sorted({n.split("/")[-1] for n in names})
        except Exception:
            pass

    if not maps:
        # Fallback to local maps directory if server is not reachable immediately
        maps_dir = os.path.join(HERE, "CarlaUE4", "Content", "Carla", "Maps")
        if os.path.isdir(maps_dir):
            for item in os.listdir(maps_dir):
                if item.endswith(".umap") and not item.startswith("Annotation"):
                    maps.append(item[:-5])
            maps = sorted(set(maps))

    if not maps:
        sys.exit(f"Could not reach CARLA at {host}:{port} and no local maps found. Is the server running?")

    if not include_opt:
        maps = [m for m in maps if "_Opt" not in m]
    return maps


def print_maps(maps):
    for i, m in enumerate(maps, 1):
        note = "  (large map, heavy on GPU memory)" if m in LARGE_MAPS else ""
        print(f"  {i:2d}. {m}{note}")


def resolve_names(names, maps):
    lookup = {m.lower(): m for m in maps}
    # Also support friendly aliases like town10 -> Town10HD, town1 -> Town01
    aliases = {
        "town10": "Town10HD",
        "town10_opt": "Town10HD_Opt",
        "town1": "Town01",
        "town2": "Town02",
        "town3": "Town03",
        "town4": "Town04",
        "town5": "Town05",
        "town6": "Town06",
        "town7": "Town07",
    }
    for alias, target in aliases.items():
        if alias not in lookup and target in maps:
            lookup[alias] = target

    chosen = []
    for n in names:
        nl = n.lower()
        if nl in ("all", "a"):
            return list(maps)
        if nl in lookup:
            matched = lookup[nl]
        else:
            # Fuzzy prefix / contains match
            matches = [m for m in maps if m.lower().startswith(nl) or nl in m.lower()]
            if matches:
                matched = matches[0]
            else:
                sys.exit(f"Unknown map '{n}'. Run with --list to see what's available.")

        if matched not in chosen:
            chosen.append(matched)
    return chosen


def parse_selection(text, maps):
    """Accepts: all | 1,3,5 | 2-4 | 1 3 5 | mixed."""
    text = text.strip().lower()
    if text in ("all", "a"):
        return list(maps)
    chosen = []
    for part in text.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices = range(int(lo), int(hi) + 1)
        else:
            indices = [int(part)]
        for i in indices:
            if not 1 <= i <= len(maps):
                raise ValueError(f"{i} is not in the list")
            if maps[i - 1] not in chosen:
                chosen.append(maps[i - 1])
    if not chosen:
        raise ValueError("nothing selected")
    return chosen


def interactive_pick(maps):
    print("\nAvailable maps:")
    print_maps(maps)
    print("\nEnter numbers (e.g. 1,3  or  2-4  or  all). Press Ctrl+C to quit.")
    while True:
        try:
            return parse_selection(input("Maps to collect from: "), maps)
        except ValueError as e:
            print(f"  Invalid choice: {e}. Try again.")


def manifest_to_labels_csv(town_dir):
    """Convert manifest.jsonl from collect_data.py to labels.csv if needed."""
    manifest_path = os.path.join(town_dir, "manifest.jsonl")
    csv_path = os.path.join(town_dir, "labels.csv")

    if not os.path.isfile(manifest_path):
        return csv_path if os.path.isfile(csv_path) else None

    # Read manifest rows and generate labels.csv
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Flatten weather dict if present
            flat = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}_{sub_k}"] = sub_v
                else:
                    flat[k] = v
            rows.append(flat)

    if not rows:
        return None

    # Write labels.csv
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def merge_labels(out_root, towns):
    """Merge per-town labels into all_labels.csv with an extra 'town' column."""
    merged_path = os.path.join(out_root, "all_labels.csv")
    merged_manifest = os.path.join(out_root, "manifest.jsonl")
    writer = None
    rows = 0

    # Ensure parent output directory exists
    os.makedirs(out_root, exist_ok=True)

    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        for town in towns:
            town_dir = os.path.join(out_root, town)
            # Ensure labels.csv exists (from manifest or direct)
            path = manifest_to_labels_csv(town_dir)
            if not path or not os.path.exists(path):
                continue
            with open(path, newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=["town"] + reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    row["town"] = town
                    writer.writerow(row)
                    rows += 1

    # Also merge manifest.jsonl if present for multimodal training compatibility
    manifest_rows = 0
    with open(merged_manifest, "w", encoding="utf-8") as mf_out:
        for town in towns:
            m_path = os.path.join(out_root, town, "manifest.jsonl")
            if os.path.isfile(m_path):
                with open(m_path, "r", encoding="utf-8") as mf_in:
                    for line in mf_in:
                        if line.strip():
                            mf_out.write(line)
                            manifest_rows += 1

    return merged_path, rows


def main():
    ap = argparse.ArgumentParser(description="Collect CARLA data from several towns")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--towns", nargs="+", help="map names, or 'all'")
    ap.add_argument("--list", action="store_true", help="list maps and exit")
    ap.add_argument("--include_opt", action="store_true",
                    help="also show the *_Opt (layered) map variants")
    ap.add_argument("--frames", type=int, default=5000, help="frames to save PER town")
    ap.add_argument("--save_every", type=int, default=2)
    ap.add_argument("--npcs", type=int, default=40)
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    if not os.path.exists(COLLECT):
        sys.exit(f"collect_data.py not found next to this script ({COLLECT})")

    maps = get_maps(args.host, args.port, args.include_opt)
    if args.list:
        print("Maps on the server:")
        print_maps(maps)
        return

    if args.towns:
        towns = resolve_names(args.towns, maps)
    else:
        towns = interactive_pick(maps)
        print(f"\nWill collect {args.frames} frames from each of: {', '.join(towns)}")
        if input("Start? [Y/n] ").strip().lower() in ("n", "no"):
            return

    # Calculate vehicles vs walkers from npcs
    vehicles = max(2, int(args.npcs * 0.7))
    walkers = max(1, int(args.npcs * 0.3))

    done, failed = [], []
    try:
        for n, town in enumerate(towns, 1):
            print(f"\n=== [{n}/{len(towns)}] {town} ===")
            if town in LARGE_MAPS:
                print("Note: large map. If it crashes, lower --npcs or pick a smaller town.")
            town_out = os.path.join(args.out, town)
            cmd = [
                sys.executable, COLLECT,
                "--host", args.host,
                "--port", str(args.port),
                "--towns", town,
                "--samples-per-episode", str(args.frames),
                "--episodes-per-town", "1",
                "--vehicles", str(vehicles),
                "--walkers", str(walkers),
                "--out", town_out,
            ]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                # Convert manifest to labels.csv
                manifest_to_labels_csv(town_out)
                done.append(town)
            else:
                failed.append(town)
            time.sleep(3)  # give the server a moment before loading the next map
    except KeyboardInterrupt:
        print("\nInterrupted. Merging what was collected so far...")

    merged, rows = merge_labels(args.out, towns)
    print("\n=== Summary ===")
    print(f"Finished: {', '.join(done) or 'none'}")
    if failed:
        print(f"Failed:   {', '.join(failed)}")
    print(f"Combined labels: {merged} ({rows} rows)")
    print(f"Image path for a row: {args.out}/<town>/<episode_id>/rgb/<id>.jpg")


if __name__ == "__main__":
    main()
