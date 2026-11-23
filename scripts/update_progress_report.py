"""Update progress.pdf by prepending the latest implementation update.

Adds new milestone findings (steering bias elimination, sustained autonomous driving telemetry,
obstacle stopping diagnosis, and next-generation pathfinding architecture) to the top of
progress.pdf, while pushing older findings to the bottom.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pypdf

A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69
DPI = 300

def create_latest_update_page():
    """Render the 21 November 2026 latest implementation update page as a PDF."""
    fig = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 1. Header category
    ax.text(0.06, 0.962, "LATEST IMPLEMENTATION UPDATE  |  21 NOVEMBER 2026",
            fontsize=8.5, fontweight="bold", color="#007a87", family="sans-serif")

    # 2. Title and subtitle
    ax.text(0.06, 0.934, "Steering Model Improved & Obstacle Behavior Diagnosis",
            fontsize=14.5, fontweight="bold", color="#142d55", family="sans-serif")
    ax.text(0.06, 0.914, "Elimination of steering bias, live driving telemetry, and next-gen pathfinder architecture",
            fontsize=9.2, color="#555555", family="sans-serif")

    # Separator line
    ax.plot([0.06, 0.94], [0.904, 0.904], color="#cbd5e1", lw=0.9)

    # 3. Overview Paragraph
    overview_text = (
        "Following the navigation integration, we diagnosed and eliminated the steering model's severe directional bias\n"
        "(99.2% reduction, from +0.0613 to -0.0005), enabling sustained autonomous driving for over 760 meters and 2.5+\n"
        "minutes in CARLA Town03. However, live trials revealed a critical safety limitation: because PilotNet only outputs\n"
        "lateral steering, the car does not stop when something gets close, continuing to drive and push into obstacles."
    )
    ax.text(0.06, 0.894, overview_text, fontsize=8.0, color="#222222", family="sans-serif", va="top", linespacing=1.3)

    # -------------------------------------------------------------------------
    # SECTION 1 & 2: TWO COLUMNS (Y: 0.825 to 0.450)
    # -------------------------------------------------------------------------

    # Left Column Header & Bullets
    ax.text(0.06, 0.820, "1. LATERAL STEERING BIAS ELIMINATED", fontsize=8.5, fontweight="bold", color="#142d55")
    s1_text = (
        "• Root Cause: 975 right vs 348 left turns (2.8x skew) in raw data pushed head.bias to +0.177.\n"
        "• Symmetric Augmentation: Exact horizontal mirroring (x -> flip(x), y -> -y) doubled data to\n"
        "  8,346 frames with exact 1:1 balance (1,755 left/right; mean steer = 0.000000).\n"
        "• Anti-Bias Inference: 0.5*(f(x) - f(flip(x))) deployed in demos; cancels residual offsets."
    )
    ax.text(0.06, 0.806, s1_text, fontsize=6.8, color="#333333", va="top", linespacing=1.25)

    # Embed Figure 14 (Steering bias chart)
    fig14_path = os.path.join("report", "figures", "figure_14_steering_bias_fix.png")
    if os.path.isfile(fig14_path):
        im14 = Image.open(fig14_path)
        ax.imshow(im14, extent=[0.06, 0.485, 0.585, 0.725], aspect="auto", zorder=2)

    # Bias Comparison Mini-Table
    table_card = patches.FancyBboxPatch((0.06, 0.470), 0.425, 0.105, boxstyle="round,pad=0.008", fc="#f8fafc", ec="#cbd5e1", lw=0.8)
    ax.add_patch(table_card)
    ax.text(0.07, 0.563, "Validation Split (n=248) | Old vs New Model Comparison", fontsize=7.0, fontweight="bold", color="#142d55")
    table_text = (
        "Metric                         Old Model (Raw)    New Model (Raw)    New Model (Symmetric)\n"
        "------------------------------------------------------------------------------------------\n"
        "Net Steering Bias             +0.0613 (Right)     +0.0042 (-93%)     -0.0005 (-99.2%)\n"
        "Right Turns (> +0.02)         151 frames          73 frames          59 frames\n"
        "Left Turns (< -0.02)          20 frames           88 frames          95 frames (Balanced)\n"
        "Straight Frames               77 frames           87 frames          94 frames"
    )
    ax.text(0.07, 0.550, table_text, fontsize=5.8, family="monospace", color="#1e293b", va="top", linespacing=1.2)

    # Right Column Header & Bullets
    ax.text(0.515, 0.820, "2. TELEMETRY: SMOOTH STEER VS OBSTACLE COLLISION", fontsize=8.5, fontweight="bold", color="#142d55")
    s2_text = (
        "• Smooth Lateral Control: Run 20261119_192151 achieved 159.5 seconds and 763.0 meters\n"
        "  continuous model driving across avenues with near-zero lane boundary invasions.\n"
        "• Obstacle Collision Mechanism: Logged 155 collision ticks pushing into stopped vehicles.\n"
        "  PilotNet only controls steering (Image -> Steer); speed control is completely decoupled."
    )
    ax.text(0.515, 0.806, s2_text, fontsize=6.8, color="#333333", va="top", linespacing=1.25)

    # Embed Figure 15 (Telemetry timeline)
    fig15_path = os.path.join("report", "figures", "figure_15_driving_telemetry_limitation.png")
    if os.path.isfile(fig15_path):
        im15 = Image.open(fig15_path)
        ax.imshow(im15, extent=[0.515, 0.94, 0.585, 0.725], aspect="auto", zorder=2)

    # Telemetry explanation box
    tel_card = patches.FancyBboxPatch((0.515, 0.470), 0.425, 0.105, boxstyle="round,pad=0.008", fc="#fff7ed", ec="#fdba74", lw=0.8)
    ax.add_patch(tel_card)
    ax.text(0.525, 0.563, "Root Cause of Obstacle Pushing in Model Mode", fontsize=7.0, fontweight="bold", color="#c2410c")
    tel_desc = (
        "In live_demo.py (lines 551-554), throttle is calculated strictly as:\n"
        "    throttle = clip((target_speed - speed) * 0.04, 0.0, 0.5)\n"
        "When an obstacle halts the vehicle, speed drops to 0 km/h. Because\n"
        "speed < target (25 km/h), controller continuously outputs max throttle (0.50),\n"
        "causing repeated collisions instead of braking or re-routing."
    )
    ax.text(0.525, 0.550, tel_desc, fontsize=5.8, color="#7c2d12", va="top", linespacing=1.2)

    # -------------------------------------------------------------------------
    # SECTION 3: NEXT-GEN PATHFINDER & OBSTACLE REDIRECTION (Y: 0.450 to 0.070)
    # -------------------------------------------------------------------------
    ax.plot([0.06, 0.94], [0.455, 0.455], color="#cbd5e1", lw=0.8)
    ax.text(0.06, 0.440, "3. NEXT-GENERATION PATH FINDER & DYNAMIC OBSTACLE REDIRECTION", fontsize=9.0, fontweight="bold", color="#142d55")

    p3_text = (
        "• Current Path Finder Limitation: GlobalRoutePlanner uses static NetworkX Dijkstra routing along coarse 2.0-meter\n"
        "  chord waypoints. It has no obstacle awareness, cannot deviate from lane centerlines, and cannot plan evasive lane maneuvers.\n"
        "• Real-Life Behavior Required: When blocked, an autonomous vehicle must: (1) apply Automatic Emergency Braking (AEB),\n"
        "  (2) evaluate adjacent lane clearance, (3) dynamically redirect/overtake around the obstacle, and (4) resume the global path."
    )
    ax.text(0.06, 0.426, p3_text, fontsize=6.8, color="#333333", va="top", linespacing=1.3)

    # Embed Figure 16 (Architecture Diagram)
    fig16_path = os.path.join("report", "figures", "figure_16_nextgen_pathfinder_architecture.png")
    if os.path.isfile(fig16_path):
        im16 = Image.open(fig16_path)
        ax.imshow(im16, extent=[0.06, 0.94, 0.170, 0.360], aspect="auto", zorder=2)

    # Three-Phase Implementation Roadmap Boxes
    box_y = 0.080
    box_h = 0.075
    
    # Box 1: AEB Safety
    rect1 = patches.FancyBboxPatch((0.06, box_y), 0.275, box_h, boxstyle="round,pad=0.01", fc="#fff8e1", ec="#f59e0b", lw=1)
    ax.add_patch(rect1)
    ax.text(0.07, box_y + box_h - 0.012, "Phase 1: AEB Stop & Hold", fontsize=7.2, fontweight="bold", color="#b45309")
    b1_desc = "• Hook YOLO lead distance (d < 5m)\n• Zero throttle & apply full brake (1.0)\n• Hold until clear (d > 8m) before drive."
    ax.text(0.07, box_y + box_h - 0.025, b1_desc, fontsize=6.0, color="#333333", va="top", linespacing=1.2)

    # Box 2: Spline Smoothing
    rect2 = patches.FancyBboxPatch((0.362, box_y), 0.275, box_h, boxstyle="round,pad=0.01", fc="#f0fdf4", ec="#22c55e", lw=1)
    ax.add_patch(rect2)
    ax.text(0.372, box_y + box_h - 0.012, "Phase 2: Curvature Splines", fontsize=7.2, fontweight="bold", color="#15803d")
    b2_desc = "• Replace 2m chord waypoints with splines\n• Continuous curvature (kappa) tracking\n• Eliminates jerky cornering transitions."
    ax.text(0.372, box_y + box_h - 0.025, b2_desc, fontsize=6.0, color="#333333", va="top", linespacing=1.2)

    # Box 3: Frenet Planner
    rect3 = patches.FancyBboxPatch((0.665, box_y), 0.275, box_h, boxstyle="round,pad=0.01", fc="#eff6ff", ec="#3b82f6", lw=1)
    ax.add_patch(rect3)
    ax.text(0.675, box_y + box_h - 0.012, "Phase 3: Frenet Path Finder", fontsize=7.2, fontweight="bold", color="#1d4ed8")
    b3_desc = "• Decouple lateral d(s) and velocity s(t)\n• Quintic polynomial evasive trajectories\n• Dynamic overtake & seamless re-merge."
    ax.text(0.675, box_y + box_h - 0.025, b3_desc, fontsize=6.0, color="#333333", va="top", linespacing=1.2)

    # Footer
    ax.plot([0.06, 0.94], [0.062, 0.062], color="#cbd5e1", lw=0.6)
    footer_text = "Code: train_steer.py; live_demo.py; our_model_demo.py; navigation/fixed_route.py. Telemetry: results/live_runs.jsonl."
    ax.text(0.06, 0.046, footer_text, fontsize=6.5, color="#64748b")
    ax.text(0.94, 0.046, "1 / 7", fontsize=6.5, color="#64748b", ha="right")

    out_pdf = "latest_update_page.pdf"
    fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches=None)
    plt.close(fig)
    print(f"Generated {out_pdf}")
    return out_pdf

def merge_reports(new_page_pdf, previous_pdf, output_pdf):
    """Prepend the new implementation update to the top of previous report."""
    writer = pypdf.PdfWriter()

    # 1. Add latest update page at the top
    new_reader = pypdf.PdfReader(new_page_pdf)
    for page in new_reader.pages:
        writer.add_page(page)

    # 2. Add all older findings from previous report (pushed down)
    old_reader = pypdf.PdfReader(previous_pdf)
    # The previous report might have had 6 pages (starting from page 1: Nov 19 update, page 2: Nov 16 update, etc.)
    # When we prepend, page 1 is Nov 21, page 2 is Nov 19, pages 3-7 are Nov 16.
    # If previous_pdf already has an older version of the Nov 21 page, we take from Nov 19 page onwards.
    start_idx = 0
    if len(old_reader.pages) == 7: # if previous run already had 7 pages
        start_idx = 1 # skip previous attempt of page 1
    
    for i in range(start_idx, len(old_reader.pages)):
        writer.add_page(old_reader.pages[i])

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"Successfully updated {output_pdf} ({len(writer.pages)} total pages)")

if __name__ == "__main__":
    new_page = create_latest_update_page()
    merge_reports(new_page, "progress.pdf", "progress_updated.pdf")
    # Replace progress.pdf with progress_updated.pdf
    if os.path.isfile("progress_updated.pdf"):
        if os.path.isfile("progress.pdf"):
            os.remove("progress.pdf")
        os.rename("progress_updated.pdf", "progress.pdf")
    if os.path.isfile("latest_update_page.pdf"):
        os.remove("latest_update_page.pdf")
