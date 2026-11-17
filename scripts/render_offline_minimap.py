"""Pure Python Offline Minimap Renderer (No CARLA or OpenCV required).

Renders a 2D top-down view of Town03 with all 10 checkpoints (P1-P10)
and the ego vehicle, and saves it as a 24-bit standard BMP file in results/
for immediate visual inspection on development laptops without GPU/CARLA.
"""

import math
import os
import struct
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navigation.checkpoints import load_checkpoints, DEFAULT_CONFIG_PATH
from visualization.minimap import Minimap, DEFAULT_ROAD_CACHE_PATH


def write_bmp(filename: str, width: int, height: int, pixels: list):
    """Writes a 24-bit BGR BMP image file without external dependencies."""
    row_bytes = width * 3
    padding = (4 - (row_bytes % 4)) % 4
    image_size = (row_bytes + padding) * height
    file_size = 54 + image_size

    # BMP Header
    bmp_header = struct.pack(
        "<2sIHHI",
        b"BM",
        file_size,
        0,
        0,
        54,
    )
    # DIB Header (BITMAPINFOHEADER)
    dib_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,  # positive height = bottom-to-top
        1,
        24,
        0,
        image_size,
        2835,
        2835,
        0,
        0,
    )

    with open(filename, "wb") as f:
        f.write(bmp_header)
        f.write(dib_header)
        # BMP stores rows from bottom to top
        for y in range(height - 1, -1, -1):
            row_data = bytearray()
            for x in range(width):
                b, g, r = pixels[y][x]
                row_data.extend((b, g, r))
            row_data.extend(b"\x00" * padding)
            f.write(row_data)


def render_offline_preview(
    output_path: str = os.path.join(PROJECT_ROOT, "results", "offline_minimap_preview.bmp"),
    size: int = 400,
):
    print(f"[render] Generating offline Town03 minimap preview ({size}x{size})...")
    checkpoints = load_checkpoints(DEFAULT_CONFIG_PATH)
    minimap = Minimap(
        size=size,
        checkpoints=checkpoints,
        road_cache_path=DEFAULT_ROAD_CACHE_PATH,
        padding=24,
    )

    # Create pixel buffer initialized to dark charcoal background
    # BGR format: (B, G, R)
    bg_color = (30, 30, 30)
    road_color = (175, 175, 175)
    cp_color = (0, 130, 245)      # Red-orange
    ego_color = (0, 235, 0)       # Green
    white = (255, 255, 255)

    pixels = [[bg_color for _ in range(size)] for _ in range(size)]

    def draw_pixel(x, y, color):
        if 0 <= x < size and 0 <= y < size:
            pixels[y][x] = color

    def draw_line(x0, y0, x1, y1, color, thickness=1):
        """Bresenham line algorithm with thickness."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for tx in range(-thickness // 2, thickness // 2 + 1):
                for ty in range(-thickness // 2, thickness // 2 + 1):
                    draw_pixel(x0 + tx, y0 + ty, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_circle(cx, cy, r, color, fill=True):
        for y in range(max(0, cy - r), min(size, cy + r + 1)):
            for x in range(max(0, cx - r), min(size, cx + r + 1)):
                d_sq = (x - cx) ** 2 + (y - cy) ** 2
                if fill and d_sq <= r ** 2:
                    pixels[y][x] = color
                elif not fill and abs(d_sq - r ** 2) <= r:
                    pixels[y][x] = color

    # 1. Draw Road Network
    for segment in minimap.roads:
        for i in range(len(segment) - 1):
            px0, py0 = minimap.world_to_pixel(segment[i][0], segment[i][1])
            px1, py1 = minimap.world_to_pixel(segment[i + 1][0], segment[i + 1][1])
            draw_line(px0, py0, px1, py1, road_color, thickness=2)

    # 2. Draw Checkpoints P1 to P10
    for cp in checkpoints:
        px, py = minimap.world_to_pixel(cp.x, cp.y)
        # Dark outline then orange dot
        draw_circle(px, py, 5, (15, 15, 15), fill=True)
        draw_circle(px, py, 4, cp_color, fill=True)

    # 3. Draw Ego Vehicle at roundabout near P1, heading North (yaw = 90)
    ego_x, ego_y, ego_yaw = 142.1, 195.4, 90.0
    ev_px, ev_py = minimap.world_to_pixel(ego_x, ego_y)
    dx, dy = minimap.yaw_to_direction(ego_yaw)
    nx, ny = -dy, dx

    # Draw directional triangle
    tip_x, tip_y = int(ev_px + dx * 12), int(ev_py + dy * 12)
    left_x, left_y = int(ev_px - dx * 6 + nx * 6), int(ev_py - dy * 6 + ny * 6)
    right_x, right_y = int(ev_px - dx * 6 - nx * 6), int(ev_py - dy * 6 - ny * 6)

    draw_line(tip_x, tip_y, left_x, left_y, ego_color, thickness=2)
    draw_line(left_x, left_y, right_x, right_y, ego_color, thickness=2)
    draw_line(right_x, right_y, tip_x, tip_y, ego_color, thickness=2)
    draw_circle(ev_px, ev_py, 2, white, fill=True)

    # 4. Save BMP
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_bmp(output_path, size, size, pixels)
    print(f"[success] Offline minimap preview saved to: {output_path}")
    print(f"          Visual verification: Open '{output_path}' in any image viewer!")


if __name__ == "__main__":
    render_offline_preview()

