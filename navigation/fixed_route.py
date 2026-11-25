"""Fixed Town03 checkpoint route helpers for model demos."""

import sys
from pathlib import Path

from navigation.checkpoints import DEFAULT_CONFIG_PATH, load_checkpoints


ROUTE_TOWN = "Town03"
ROUTE_START_SPAWN_INDEX = 0
ROUTE_CHECKPOINT_IDS = (
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"
)
ROUTE_RESOLUTION_M = 2.0
ROUTE_MAX_DISTANCE_M = 3000.0
MODEL_STEER_WEIGHT = 0.25


def build_route_from_locations(planner, start_location, destinations):
    import networkx as nx

    route = []
    route_distance = 0.0
    previous_location = None
    previous_label = f"spawn {ROUTE_START_SPAWN_INDEX}"
    current_location = start_location
    route_limit_reached = False

    for destination_label, destination in destinations:
        try:
            segment = planner.trace_route(current_location, destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise RuntimeError(
                f"Could not plan route from {previous_label} to {destination_label}"
            ) from exc
        if not segment:
            raise RuntimeError(
                f"Route planner returned no waypoints for {destination_label}"
            )

        for waypoint, road_option in segment:
            location = waypoint.transform.location
            if previous_location is not None:
                step_distance = previous_location.distance(location)
                if step_distance < 0.05:
                    continue
                if route_distance + step_distance > ROUTE_MAX_DISTANCE_M:
                    route_limit_reached = True
                    break
                route_distance += step_distance
            route.append((waypoint, road_option))
            previous_location = location

        if route_limit_reached or route_distance >= ROUTE_MAX_DISTANCE_M:
            break
        current_location = destination
        previous_label = destination_label

    if not route:
        raise RuntimeError("The fixed checkpoint route contains no waypoints")
    return route, route_distance


def build_fixed_route(world_map, checkpoints_path=None):
    api_path = Path(__file__).resolve().parents[1] / "PythonAPI" / "carla"
    if not api_path.is_dir():
        raise RuntimeError(f"CARLA navigation API not found: {api_path}")
    if str(api_path) not in sys.path:
        sys.path.insert(0, str(api_path))

    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    spawn_points = world_map.get_spawn_points()
    if ROUTE_START_SPAWN_INDEX >= len(spawn_points):
        raise RuntimeError(
            f"{ROUTE_TOWN} has {len(spawn_points)} spawn points; "
            f"fixed start index {ROUTE_START_SPAWN_INDEX} is invalid"
        )

    planner = GlobalRoutePlanner(world_map, ROUTE_RESOLUTION_M)
    checkpoint_path = checkpoints_path or DEFAULT_CONFIG_PATH
    checkpoints = {
        checkpoint.id: checkpoint
        for checkpoint in load_checkpoints(checkpoint_path)
    }
    missing_ids = [
        checkpoint_id
        for checkpoint_id in ROUTE_CHECKPOINT_IDS
        if checkpoint_id not in checkpoints
    ]
    if missing_ids:
        raise RuntimeError(
            f"Checkpoint config is missing fixed route targets: {', '.join(missing_ids)}"
        )
    destinations = [
        (
            checkpoint_id,
            carla.Location(
                x=checkpoints[checkpoint_id].x,
                y=checkpoints[checkpoint_id].y,
                z=checkpoints[checkpoint_id].z,
            ),
        )
        for checkpoint_id in ROUTE_CHECKPOINT_IDS
    ]
    route, route_distance = build_route_from_locations(
        planner, spawn_points[ROUTE_START_SPAWN_INDEX].location, destinations
    )
    return route, route_distance, planner, spawn_points


def build_navigation_agent(vehicle, world_map, planner, route, target_speed):
    from agents.navigation.basic_agent import BasicAgent

    agent = BasicAgent(
        vehicle,
        target_speed=target_speed,
        opt_dict={"sampling_resolution": ROUTE_RESOLUTION_M},
        map_inst=world_map,
        grp_inst=planner,
    )
    agent.set_global_plan(route, stop_waypoint_creation=True)
    return agent


def blend_model_and_navigation_steering(model_steer, navigation_steer):
    return float(max(
        -1.0,
        min(
            1.0,
            MODEL_STEER_WEIGHT * model_steer
            + (1.0 - MODEL_STEER_WEIGHT) * navigation_steer,
        ),
    ))
