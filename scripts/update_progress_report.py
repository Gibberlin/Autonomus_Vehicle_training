"""Update progress.pdf by prepending the latest implementation updates.

Generates:
- Page 1: 26 November 2026 Milestone Update (Autonomous Route Following, Minimap Path Tracing & Safety Integration)
- Page 2: 21 November 2026 Milestone Update (Steering Bias Fix & Obstacle Pushing Diagnosis)
- Page 3: 19 November 2026 Update (Fixed-Route Navigation)
- Pages 4-8: Comprehensive baseline and model report (Figures 1-13, benchmarks)
Total: 8 pages.
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


def create_26nov_milestone_page():
    """Render the 26 November 2026 Milestone Update page as a PDF."""
    fig = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 1. Header category
    ax.text(0.06, 0.962, "LATEST IMPLEMENTATION UPDATE  |  26 NOVEMBER 2026",
            fontsize=8.5, fontweight="bold", color="#007a87", family="sans-serif")

    # 2. Title and subtitle
    ax.text(0.06, 0.934, "Autonomous Route Following, Minimap Tracing & Safety Integration",
            fontsize=14.0, fontweight="bold", color="#0f172a", family="sans-serif")
    ax.text(0.06, 0.914, "End-to-end destination routing, high-contrast minimap polyline, Stanley route tracking, and front-zone AEB",
            fontsize=8.8, color="#475569", family="sans-serif")

    # Separator line
    ax.plot([0.06, 0.94], [0.904, 0.904], color="#cbd5e1", lw=0.9)

    # 3. Overview Paragraph
    overview_text = (
        "We completed the end-to-end integration connecting destination selection, OpenDRIVE road graph path planning,\n"
        "high-contrast minimap polyline rendering, and autonomous vehicle control. Using CARLA's OpenDRIVE topology with\n"
        "heading-aligned road snapping, the vehicle computes legal road routes, renders high-contrast polylines on the\n"
        "minimap, follows waypoints with speed-adaptive lookahead, and decelerates to a complete stop inside a 2.5m arrival\n"
        "tolerance while obeying front and front-side emergency braking with rear traffic false-brake rejection."
    )
    ax.text(0.06, 0.894, overview_text, fontsize=7.6, color="#1e293b", family="sans-serif", va="top", linespacing=1.28)

    # -------------------------------------------------------------------------
    # SECTION 1: ARCHITECTURE FLOWCHART (Figure 17)
    # -------------------------------------------------------------------------
    ax.text(0.06, 0.816, "1. END-TO-END AUTONOMOUS NAVIGATION PIPELINE",
            fontsize=8.5, fontweight="bold", color="#0f172a")

    fig17_path = os.path.join("report", "figures", "figure_17_autonomous_navigation_pipeline.png")
    if os.path.isfile(fig17_path):
        im17 = Image.open(fig17_path)
        # Aspect ratio 2.417:1 -> height = 0.88 * 8.27 / 2.417 / 11.69 = 0.2576
        y_top_17 = 0.802
        y_bot_17 = y_top_17 - 0.2576
        ax.imshow(im17, extent=[0.06, 0.94, y_bot_17, y_top_17], aspect="auto", zorder=2)

    # -------------------------------------------------------------------------
    # SECTION 2: VERIFICATION PANELS (Figure 18)
    # -------------------------------------------------------------------------
    ax.text(0.06, 0.528, "2. TRACKING CONTROLLER, MINIMAP HUD & SAFETY CHARACTERIZATION",
            fontsize=8.5, fontweight="bold", color="#0f172a")

    fig18_path = os.path.join("report", "figures", "figure_18_navigation_tracking_and_safety.png")
    if os.path.isfile(fig18_path):
        im18 = Image.open(fig18_path)
        # Aspect ratio 2.586:1 -> height = 0.88 * 8.27 / 2.586 / 11.69 = 0.2407
        y_top_18 = 0.514
        y_bot_18 = y_top_18 - 0.2407
        ax.imshow(im18, extent=[0.06, 0.94, y_bot_18, y_top_18], aspect="auto", zorder=2)

    # -------------------------------------------------------------------------
    # SECTION 3: SUBSYSTEM COMPARISON TABLE
    # -------------------------------------------------------------------------
    card_y = 0.075
    card_h = 0.182
    table_card = patches.FancyBboxPatch((0.06, card_y), 0.88, card_h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                        fc="#f8fafc", ec="#94a3b8", lw=0.9)
    ax.add_patch(table_card)

    # Table Card Header Pill
    hdr_pill = patches.FancyBboxPatch((0.068, card_y + card_h - 0.028), 0.864, 0.022,
                                      boxstyle="round,pad=0.004,rounding_size=0.008",
                                      fc="#e2e8f0", ec="none")
    ax.add_patch(hdr_pill)
    ax.text(0.080, card_y + card_h - 0.017, "Autonomous Driving Subsystem Comparison: Pre-Nov 26 vs Nov 26 Milestone Update",
            fontsize=7.6, fontweight="bold", color="#0f172a", va="center")

    table_content = (
        "Capability / Subsystem             Previous Implementation              Nov 26 Milestone Update                   Status\n"
        "--------------------------------------------------------------------------------------------------------------------------------\n"
        "Destination-to-Route Planning     Disconnected / None                  RoadRouter (Universal OpenDRIVE Topology) VERIFIED & TESTED\n"
        "Minimap Route Visualization       Markers only, no route polyline      High-Contrast Cyan Polyline + Traversed   INTEGRATED & LIVE\n"
        "Lateral Tracking Control          Unbounded circling loop (30-130 std) Stanley + Pure Pursuit + Yaw bounded      RESOLVED (0 CIRCLING)\n"
        "Obstacle Safety Zone              Full 360-deg false brake lockups     Front & Front-Sides Only (Rear Ignored)   VERIFIED (0 FALSE STOPS)\n"
        "Arrival Deceleration & Halt       None (drove indefinitely past target)20m speed taper; <2.5m full brake stop    ACTIVE & VERIFIED\n"
        "Vehicle Control Modes             Overrides & conflicts                Clean m (model/route), a (auto), d (dest) RESTORED & SMOOTH\n"
        "Offline Unit Test Suite           33 unit tests                        40 passing tests in test suite            100% PASS (1.72s)"
    )
    ax.text(0.078, card_y + card_h - 0.038, table_content, fontsize=5.3, family="monospace", color="#0f172a", va="top", linespacing=1.28)

    # Footer
    ax.plot([0.06, 0.94], [0.060, 0.060], color="#cbd5e1", lw=0.6)
    footer_text = "Modules: navigation/router.py; navigation/route_follower.py; visualization/minimap.py; live_demo.py; tests/test_router_and_follower.py."
    ax.text(0.06, 0.044, footer_text, fontsize=6.2, color="#64748b")
    ax.text(0.94, 0.044, "1 / 8", fontsize=6.5, color="#64748b", ha="right")

    out_pdf = "page_01_26nov.pdf"
    fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches=None)
    fig.savefig(os.path.join("report", "page_01_preview.png"), dpi=150, bbox_inches=None)
    plt.close(fig)
    print(f"Generated {out_pdf} and report/page_01_preview.png")
    return out_pdf


def create_21nov_update_page():
    """Render the 21 November 2026 latest implementation update page as a PDF."""
    fig = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 1. Header category
    ax.text(0.06, 0.962, "IMPLEMENTATION UPDATE  |  21 NOVEMBER 2026",
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
        "• Root Cause: 975 right vs 348 left turns (2.8x skew)\n"
        "  in raw data pushed head.bias to +0.177.\n"
        "• Symmetric Augmentation: Exact mirroring (x -> flip(x),\n"
        "  y -> -y) doubled data to 8,346 frames with exact\n"
        "  1:1 balance (1,755 left/right; mean steer = 0.0000).\n"
        "• Anti-Bias Inference: 0.5*(f(x) - f(flip(x))) deployed\n"
        "  in demos; cancels residual offsets."
    )
    ax.text(0.06, 0.806, s1_text, fontsize=6.5, color="#333333", va="top", linespacing=1.22)

    # Embed Figure 14 (Steering bias chart)
    fig14_path = os.path.join("report", "figures", "figure_14_steering_bias_fix.png")
    if os.path.isfile(fig14_path):
        im14 = Image.open(fig14_path)
        ax.imshow(im14, extent=[0.06, 0.485, 0.585, 0.725], aspect="auto", zorder=2)

    # Bias Comparison Mini-Table
    table_card = patches.FancyBboxPatch((0.06, 0.468), 0.425, 0.108, boxstyle="round,pad=0.008", fc="#f8fafc", ec="#cbd5e1", lw=0.8)
    ax.add_patch(table_card)
    ax.text(0.07, 0.563, "Validation Split (n=248) | Old vs New Model", fontsize=6.5, fontweight="bold", color="#142d55")
    table_text = (
        "Metric             Old (Raw)    New (Raw)    New (Sym)\n"
        "------------------------------------------------------\n"
        "Net Steering Bias  +0.0613      +0.0042      -0.0005\n"
        "Right Turns (>0.02)151 frames   73 frames    59 frames\n"
        "Left Turns (<-0.02) 20 frames   88 frames    95 frames (Bal)\n"
        "Straight Frames     77 frames   87 frames    94 frames"
    )
    ax.text(0.07, 0.548, table_text, fontsize=4.9, family="monospace", color="#1e293b", va="top", linespacing=1.22)

    # Right Column Header & Bullets
    ax.text(0.515, 0.820, "2. TELEMETRY: SMOOTH STEER VS OBSTACLE COLLISION", fontsize=8.5, fontweight="bold", color="#142d55")
    s2_text = (
        "• Smooth Lateral Control: Run 20261119_192151 achieved\n"
        "  159.5 s and 763.0 m continuous model driving across\n"
        "  avenues with near-zero lane boundary invasions.\n"
        "• Obstacle Collision Mechanism: Logged 155 collision\n"
        "  ticks pushing into stopped vehicles. PilotNet only\n"
        "  controls steer; speed control is completely decoupled."
    )
    ax.text(0.515, 0.806, s2_text, fontsize=6.5, color="#333333", va="top", linespacing=1.22)

    # Embed Figure 15 (Telemetry timeline)
    fig15_path = os.path.join("report", "figures", "figure_15_driving_telemetry_limitation.png")
    if os.path.isfile(fig15_path):
        im15 = Image.open(fig15_path)
        ax.imshow(im15, extent=[0.515, 0.94, 0.585, 0.725], aspect="auto", zorder=2)

    # Telemetry explanation box
    tel_card = patches.FancyBboxPatch((0.515, 0.468), 0.425, 0.108, boxstyle="round,pad=0.008", fc="#fff7ed", ec="#fdba74", lw=0.8)
    ax.add_patch(tel_card)
    ax.text(0.525, 0.563, "Root Cause of Obstacle Pushing in Model Mode", fontsize=6.5, fontweight="bold", color="#c2410c")
    tel_desc = (
        "In live_demo.py, throttle is calculated strictly as:\n"
        "    throttle = clip((target_speed - speed) * 0.04, 0.0, 0.5)\n"
        "When an obstacle halts the vehicle, speed drops to 0 km/h.\n"
        "Because speed < target (25 km/h), controller continuously\n"
        "outputs max throttle, causing repeated collisions\n"
        "instead of braking or safely stopping."
    )
    ax.text(0.525, 0.548, tel_desc, fontsize=5.0, color="#7c2d12", va="top", linespacing=1.22)

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
    ax.text(0.94, 0.046, "2 / 8", fontsize=6.5, color="#64748b", ha="right")

    out_pdf = "page_02_21nov.pdf"
    fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches=None)
    fig.savefig(os.path.join("report", "page_02_preview.png"), dpi=150, bbox_inches=None)
    plt.close(fig)
    print(f"Generated {out_pdf} and report/page_02_preview.png")
    return out_pdf


def build_progress_report():
    """Assemble all 8 pages of the comprehensive progress report."""
    page1_pdf = create_26nov_milestone_page()
    page2_pdf = create_21nov_update_page()

    writer = pypdf.PdfWriter()

    # 1. Page 1: 26 Nov 2026 Milestone Update
    r1 = pypdf.PdfReader(page1_pdf)
    writer.add_page(r1.pages[0])

    # 2. Page 2: 21 Nov 2026 Milestone Update
    r2 = pypdf.PdfReader(page2_pdf)
    writer.add_page(r2.pages[0])

    # 3. Pages 3-8: 19 Nov update + Original 5 report pages from progress.pdf
    # Read existing progress.pdf to extract 19 Nov update and original pages
    cur_reader = pypdf.PdfReader("progress.pdf")
    # In current 7-page progress.pdf:
    # Page 0 was 21 Nov
    # Page 1 was 19 Nov
    # Pages 2-6 were original 5 pages (Figures 1-13)
    start_from = 1 if len(cur_reader.pages) == 7 else 2
    for i in range(start_from, len(cur_reader.pages)):
        writer.add_page(cur_reader.pages[i])

    out_file = "progress_updated.pdf"
    with open(out_file, "wb") as f:
        writer.write(f)

    print(f"Generated {out_file} with {len(writer.pages)} total pages.")

    # Safely replace progress.pdf
    if os.path.isfile("progress.pdf"):
        os.remove("progress.pdf")
    os.rename(out_file, "progress.pdf")

    # Clean temporary pdf files
    for temp in [page1_pdf, page2_pdf]:
        if os.path.isfile(temp):
            os.remove(temp)

    print("Successfully replaced progress.pdf!")


if __name__ == "__main__":
    build_progress_report()
