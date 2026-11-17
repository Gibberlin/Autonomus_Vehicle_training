"""Town03 2D Live Minimap with Checkpoints and Vehicle Tracking.

Renders a top-down, clean 2D representation of the CARLA Town03 road network,
overlays 10 fixed checkpoints (P1 to P10), and tracks the ego vehicle in real time
with directional heading on top of OpenCV camera frames.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from navigation.checkpoints import (
    DEFAULT_CONFIG_PATH,
    Checkpoint,
    load_checkpoints,
)

DEFAULT_ROAD_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "town03_roads_cache.json",
)

# Color definitions in BGR format (for OpenCV)
COLOR_BACKGROUND = (30, 30, 30)       # Dark charcoal
COLOR_PANEL_BORDER = (80, 80, 80)     # Subtle border
COLOR_ROAD = (175, 175, 175)          # Light grey
COLOR_ROAD_OUTLINE = (100, 100, 100)  # Medium grey road casing
COLOR_CHECKPOINT = (0, 130, 245)      # Red-orange (P1-P10)
COLOR_CHECKPOINT_BORDER = (20, 20, 20)# Dark contrast outline
COLOR_VEHICLE = (0, 235, 0)           # Bright green
COLOR_VEHICLE_BORDER = (0, 60, 0)     # Dark green outline
COLOR_TEXT = (255, 255, 255)          # White
COLOR_LEGEND_BG = (20, 20, 20)        # Legend banner


class Minimap:
    """Town03 Top-Down 2D Minimap Renderer and Overlay."""

    def __init__(
        self,
        size: int = 260,
        position: str = "top-right",
        opacity: float = 0.85,
        carla_map: Optional[Any] = None,
        checkpoints: Optional[List[Checkpoint]] = None,
        checkpoints_path: str = DEFAULT_CONFIG_PATH,
        road_cache_path: str = DEFAULT_ROAD_CACHE_PATH,
        padding: int = 14,
    ):
        """Initializes the Minimap.

        Args:
            size: Square dimensions (width & height in pixels) of the minimap panel.
            position: Position on the camera frame ('top-right' or 'top-left').
            opacity: Transparency blend weight (0.0 to 1.0).
            carla_map: Optional live CARLA map instance (for extracting road network).
            checkpoints: Optional list of Checkpoint objects.
            checkpoints_path: Path to load checkpoints JSON if not provided.
            road_cache_path: Path to cached Town03 road polylines.
            padding: Inner margin in pixels between roads and minimap border.
        """
        self.size = max(80, int(size))
        self.position = position.lower()
        if self.position not in ("top-right", "top-left"):
            self.position = "top-right"
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.padding = padding
        self.road_cache_path = road_cache_path

        # 1. Load or extract checkpoints
        if checkpoints is not None:
            self.checkpoints = checkpoints
        elif os.path.isfile(checkpoints_path):
            self.checkpoints = load_checkpoints(checkpoints_path)
        else:
            self.checkpoints = []

        # 2. Load or extract road polylines
        self.roads: List[List[Tuple[float, float]]] = []
        self.min_x: float = -260.0
        self.max_x: float = 260.0
        self.min_y: float = -260.0
        self.max_y: float = 260.0

        if carla_map is not None:
            try:
                self._extract_from_carla_map(carla_map)
            except Exception as exc:
                print(f"[minimap] Warning: Could not extract from CARLA map ({exc}); falling back to cache")
                self._load_from_cache_or_default()
        else:
            self._load_from_cache_or_default()

        # 3. Compute coordinate transformation parameters
        self._compute_transform_parameters()

        # 4. Render and cache static base image (background, roads, checkpoints, legend)
        self.base_image: Optional[np.ndarray] = None
        if np is not None and cv2 is not None:
            self._render_static_base()

    def _extract_from_carla_map(self, carla_map: Any, step_m: float = 2.0) -> None:
        """Extracts road polylines directly from CARLA map waypoints and caches them."""
        import carla

        waypoints = carla_map.generate_waypoints(step_m)
        roads: List[List[Tuple[float, float]]] = []

        for wp in waypoints:
            if getattr(wp, "lane_type", None) != carla.LaneType.Driving:
                continue
            loc1 = wp.transform.location
            for nxt in wp.next(step_m):
                loc2 = nxt.transform.location
                roads.append([(loc1.x, loc1.y), (loc2.x, loc2.y)])

        if not roads:
            raise RuntimeError("CARLA map returned no road waypoints")

        self.roads = roads
        all_x = [pt[0] for segment in roads for pt in segment]
        all_y = [pt[1] for segment in roads for pt in segment]
        self.min_x = min(all_x) - 10.0
        self.max_x = max(all_x) + 10.0
        self.min_y = min(all_y) - 10.0
        self.max_y = max(all_y) + 10.0

        # Save to cache for future fast loading
        self._save_cache()

    def _save_cache(self) -> None:
        """Saves current road segments and bounds to cache file."""
        if not self.roads:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.road_cache_path)), exist_ok=True)
        data = {
            "map": "Town03",
            "bounds": {
                "min_x": round(self.min_x, 1),
                "max_x": round(self.max_x, 1),
                "min_y": round(self.min_y, 1),
                "max_y": round(self.max_y, 1),
            },
            "roads_count": len(self.roads),
            "roads": [[[round(pt[0], 2), round(pt[1], 2)] for pt in seg] for seg in self.roads],
        }
        with open(self.road_cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load_from_cache_or_default(self) -> None:
        """Loads road lines from the JSON cache file if available, or uses default bounds."""
        if os.path.isfile(self.road_cache_path):
            try:
                with open(self.road_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_roads = data.get("roads", [])
                self.roads = [[(pt[0], pt[1]) for pt in seg] for seg in raw_roads]
                bounds = data.get("bounds", {})
                self.min_x = float(bounds.get("min_x", -250.0))
                self.max_x = float(bounds.get("max_x", 250.0))
                self.min_y = float(bounds.get("min_y", -250.0))
                self.max_y = float(bounds.get("max_y", 250.0))
                return
            except Exception as exc:
                print(f"[minimap] Warning: Could not read road cache ({exc})")

        # Fallback bounds from checkpoints if no cache exists
        if self.checkpoints:
            xs = [cp.x for cp in self.checkpoints]
            ys = [cp.y for cp in self.checkpoints]
            self.min_x = min(xs) - 40.0
            self.max_x = max(xs) + 40.0
            self.min_y = min(ys) - 40.0
            self.max_y = max(ys) + 40.0

    def _compute_transform_parameters(self) -> None:
        """Computes isotropic scaling and centering offsets for world-to-pixel mapping.

        Maintains exact aspect ratio so roundabouts remain circular and road geometry
        is not distorted. Inverted vertical image axis is preserved.
        """
        drawable_w = max(10, self.size - 2 * self.padding)
        drawable_h = max(10, self.size - 2 * self.padding)

        world_span_x = max(1.0, self.max_x - self.min_x)
        world_span_y = max(1.0, self.max_y - self.min_y)

        # Uniform scale factor to prevent spatial distortion
        self.scale = min(drawable_w / world_span_x, drawable_h / world_span_y)

        rendered_w = world_span_x * self.scale
        rendered_h = world_span_y * self.scale

        # Centering offsets
        self.offset_x = self.padding + (drawable_w - rendered_w) / 2.0
        self.offset_y = self.padding + (drawable_h - rendered_h) / 2.0

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Converts CARLA world coordinates (X, Y) to 2D Minimap pixel coordinates (px, py).

        Handles the difference between CARLA's Cartesian coordinate system and 2D image
        coordinates, including the inverted vertical image axis:
        - CARLA: +X = East/Right, +Y = North/Up
        - Minimap Image: +px = Right, +py = Down (0 at top, size at bottom)
        """
        px = int(round(self.offset_x + (x - self.min_x) * self.scale))
        # Note the inverted vertical axis: (self.max_y - y)
        py = int(round(self.offset_y + (self.max_y - y) * self.scale))

        # Clamp to panel boundaries
        px = max(0, min(self.size - 1, px))
        py = max(0, min(self.size - 1, py))
        return px, py

    def yaw_to_direction(self, yaw_deg: float) -> Tuple[float, float]:
        """Converts CARLA vehicle yaw angle (degrees) to a 2D pixel direction unit vector.

        In CARLA (Unreal Engine):
        - yaw = 0 deg: vehicle faces +X (East -> Right on screen)
        - yaw = 90 deg: vehicle faces +Y (North -> Up on screen)
        - yaw = 180 / -180 deg: vehicle faces -X (West -> Left on screen)
        - yaw = -90 deg: vehicle faces -Y (South -> Down on screen)

        In Minimap pixel coordinates (with inverted vertical axis):
        - dx_pixel = cos(yaw)
        - dy_pixel = -sin(yaw)  (pointing UP when yaw = 90)
        """
        rad = math.radians(yaw_deg)
        dx_pixel = math.cos(rad)
        dy_pixel = -math.sin(rad)
        return dx_pixel, dy_pixel

    def _render_static_base(self) -> None:
        """Pre-renders the static minimap background, road network, checkpoints, and legend.

        Executed only ONCE during initialization to ensure maximum runtime performance.
        """
        if np is None or cv2 is None:
            return

        base = np.full((self.size, self.size, 3), COLOR_BACKGROUND, dtype=np.uint8)

        # 1. Draw Roads
        # First draw thicker dark casing for road edges, then light grey core
        for segment in self.roads:
            if len(segment) < 2:
                continue
            pixel_pts = [self.world_to_pixel(pt[0], pt[1]) for pt in segment]
            pts_array = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
            # Road core
            cv2.polylines(base, [pts_array], isClosed=False, color=COLOR_ROAD, thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw Checkpoints (P1 to P10)
        marker_radius = max(3, int(round(self.size * 0.016)))
        font_scale = max(0.28, self.size * 0.0012)
        font_thick = 1

        for cp in self.checkpoints:
            px, py = self.world_to_pixel(cp.x, cp.y)

            # Circular marker with dark outer border
            cv2.circle(base, (px, py), marker_radius + 1, COLOR_CHECKPOINT_BORDER, -1, lineType=cv2.LINE_AA)
            cv2.circle(base, (px, py), marker_radius, COLOR_CHECKPOINT, -1, lineType=cv2.LINE_AA)

            # Text label (e.g. "P1")
            label = cp.id
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            # Position text offset so it does not cover the marker and stays inside canvas
            text_x = px + marker_radius + 2
            text_y = py + text_h // 2
            if text_x + text_w > self.size - 4:
                text_x = px - marker_radius - text_w - 2
            if text_y - text_h < 4:
                text_y = py + marker_radius + text_h + 2
            if text_y > self.size - 18:
                text_y = self.size - 18

            # Small shadow for label readability
            cv2.putText(
                base,
                label,
                (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thick,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                base,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_TEXT,
                font_thick,
                lineType=cv2.LINE_AA,
            )

        # 3. Draw Legend at the bottom of the minimap panel
        legend_h = max(14, int(self.size * 0.065))
        legend_y1 = self.size - legend_h
        cv2.rectangle(base, (0, legend_y1), (self.size, self.size), COLOR_LEGEND_BG, -1)
        cv2.line(base, (0, legend_y1), (self.size, legend_y1), COLOR_PANEL_BORDER, 1)

        leg_font_scale = max(0.24, self.size * 0.0010)
        # Ego legend indicator (green triangle dot)
        ego_x, ego_y = int(self.size * 0.06), self.size - legend_h // 2
        cv2.circle(base, (ego_x, ego_y), 3, COLOR_VEHICLE, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            base,
            "Ego",
            (ego_x + 6, ego_y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            leg_font_scale,
            COLOR_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )

        # Checkpoints legend indicator (orange dot)
        cp_x = int(self.size * 0.45)
        cv2.circle(base, (cp_x, ego_y), 3, COLOR_CHECKPOINT, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            base,
            "P1-P10",
            (cp_x + 6, ego_y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            leg_font_scale,
            COLOR_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )

        # Panel border
        cv2.rectangle(base, (0, 0), (self.size - 1, self.size - 1), COLOR_PANEL_BORDER, 1)

        self.base_image = base

    def render(self, vehicle_x: float, vehicle_y: float, vehicle_yaw: float) -> Optional[np.ndarray]:
        """Renders the minimap with the dynamic ego vehicle marker.

        Extremely fast per-frame operation: copies pre-rendered static base image
        and draws only the vehicle directional arrow.
        """
        if self.base_image is None or np is None or cv2 is None:
            return None

        img = self.base_image.copy()

        # Convert vehicle position to pixel
        px, py = self.world_to_pixel(vehicle_x, vehicle_y)

        # Convert vehicle yaw to direction vector
        dx, dy = self.yaw_to_direction(vehicle_yaw)
        nx, ny = -dy, dx  # Normal vector (orthogonal)

        # Arrow dimensions proportional to minimap size
        length = max(6.0, self.size * 0.038)
        width = length * 0.55

        # Vehicle triangle vertices: tip, left wing, rear notch, right wing
        tip_x = px + dx * length
        tip_y = py + dy * length

        left_x = px - dx * (length * 0.5) + nx * width
        left_y = py - dy * (length * 0.5) + ny * width

        notch_x = px - dx * (length * 0.2)
        notch_y = py - dy * (length * 0.2)

        right_x = px - dx * (length * 0.5) - nx * width
        right_y = py - dy * (length * 0.5) - ny * width

        triangle_pts = np.array(
            [[tip_x, tip_y], [left_x, left_y], [notch_x, notch_y], [right_x, right_y]],
            dtype=np.int32,
        )

        # Draw filled vehicle arrow with dark contrast border
        cv2.drawContours(img, [triangle_pts], 0, COLOR_VEHICLE, -1, lineType=cv2.LINE_AA)
        cv2.drawContours(img, [triangle_pts], 0, COLOR_VEHICLE_BORDER, 1, lineType=cv2.LINE_AA)
        # Small center locator dot
        cv2.circle(img, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return img

    def overlay_on_frame(
        self,
        display_frame: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        margin_px: int = 6,
    ) -> np.ndarray:
        """Blends the rendered minimap onto the simulation display frame.

        Args:
            display_frame: BGR image from the camera / simulation pipeline.
            vehicle_x: Vehicle world coordinate X.
            vehicle_y: Vehicle world coordinate Y.
            vehicle_yaw: Vehicle world rotation yaw in degrees.
            margin_px: Padding from the edge of the display frame.

        Returns:
            display_frame with the minimap cleanly overlaid at top-right or top-left.
        """
        if self.base_image is None or cv2 is None or np is None:
            return display_frame

        minimap_img = self.render(vehicle_x, vehicle_y, vehicle_yaw)
        if minimap_img is None:
            return display_frame

        frame_h, frame_w = display_frame.shape[:2]

        # Adaptive sizing: clamp minimap size if frame is smaller than requested size
        max_allowed_w = frame_w - 2 * margin_px
        max_allowed_h = frame_h - 2 * margin_px
        max_dim = min(max_allowed_w, max_allowed_h)

        if self.size > max_dim:
            target_size = max(50, max_dim)
            scaled_minimap = cv2.resize(
                minimap_img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
        else:
            target_size = self.size
            scaled_minimap = minimap_img

        # Determine placement rectangle
        if self.position == "top-left":
            x1 = margin_px
            # Leave space for top-left status text if frame has room
            y1 = margin_px + 24 if frame_h > target_size + margin_px + 28 else margin_px
        else:  # top-right (default)
            x1 = frame_w - target_size - margin_px
            y1 = margin_px

        x2 = x1 + target_size
        y2 = y1 + target_size

        # Boundary checks
        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            x1 = max(0, min(frame_w - target_size, x1))
            y1 = max(0, min(frame_h - target_size, y1))
            x2 = x1 + target_size
            y2 = y1 + target_size

        roi = display_frame[y1:y2, x1:x2]

        # Transparent alpha blending
        if self.opacity >= 0.999:
            display_frame[y1:y2, x1:x2] = scaled_minimap
        else:
            blended = cv2.addWeighted(
                scaled_minimap, self.opacity, roi, 1.0 - self.opacity, 0.0
            )
            display_frame[y1:y2, x1:x2] = blended

        return display_frame

