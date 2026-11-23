"""Generate high-resolution report figures for model improvement analysis."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = os.path.join("report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})

def generate_figure_14():
    """Figure 14: Quantitative Comparison of Steering Bias Before and After Fix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})

    # Subplot 1: Net Steering Directional Bias
    models = ["Old Biased\nModel (Raw)", "Old Model\n(Symmetric)", "New Model\n(Raw)", "New Model\n(Symmetric)"]
    biases = [0.0613, 0.0228, 0.0042, -0.0005]
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]

    bars = ax1.bar(models, biases, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_ylabel("Net Steering Directional Bias (Pred - Truth)")
    ax1.set_title("Directional Bias Elimination")
    ax1.set_ylim(-0.015, 0.075)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, biases):
        ypos = val + 0.0025 if val >= 0 else val - 0.004
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:+.4f}",
                 ha="center", va="bottom" if val >= 0 else "top", fontweight="bold", fontsize=9)

    ax1.annotate("93% Bias Drop\n(Balanced Data)", xy=(2, 0.0042), xytext=(1.5, 0.045),
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
                 fontsize=8.5, ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#eef", ec="#88a"))

    # Subplot 2: Turn Prediction Distribution (Validation Set, n=248)
    categories = ["Right Turns\n(>+0.02)", "Straight\n(|s|<=0.02)", "Left Turns\n(<-0.02)"]
    x = np.arange(len(categories))
    width = 0.26

    old_counts = [151, 77, 20]
    new_raw_counts = [73, 87, 88]
    new_sym_counts = [59, 94, 95]

    ax2.bar(x - width, old_counts, width, label="Old Model (Raw)", color="#d9534f", edgecolor="#333")
    ax2.bar(x, new_raw_counts, width, label="New Model (Raw)", color="#5bc0de", edgecolor="#333")
    ax2.bar(x + width, new_sym_counts, width, label="New Model (Symmetric)", color="#5cb85c", edgecolor="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Number of Predicted Validation Frames")
    ax2.set_title("Prediction Symmetry & Balance (n=248)")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    for i in range(len(categories)):
        ax2.text(x[i] - width, old_counts[i] + 3, str(old_counts[i]), ha="center", fontsize=8)
        ax2.text(x[i], new_raw_counts[i] + 3, str(new_raw_counts[i]), ha="center", fontsize=8)
        ax2.text(x[i] + width, new_sym_counts[i] + 3, str(new_sym_counts[i]), ha="center", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "figure_14_steering_bias_fix.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def generate_figure_15():
    """Figure 15: Telemetry Diagnosis: Sustained Lateral Steering vs Longitudinal Collision Behavior."""
    csv_file = os.path.join("results", "live_20261119_192151.csv")
    if not os.path.isfile(csv_file):
        print(f"Warning: {csv_file} not found; generating synthetic telemetry visualization.")
        return

    df = pd.read_csv(csv_file)
    # Take first 100 seconds
    df_sub = df[(df.sim_time_s >= 20) & (df.sim_time_s <= 120)].copy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

    # Plot 1: Steering Command
    ax1.plot(df_sub.sim_time_s, df_sub.steer_command, color="#1f77b4", lw=1.5, label="PilotNet Steer Command")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax1.set_ylabel("Steer [-1, +1]")
    ax1.set_title("Live Drive Telemetry (19 Nov 2026 Run): Smooth Lateral Control vs Hazard Stopping Failure")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Plot 2: Speed and Throttle
    ax2.plot(df_sub.sim_time_s, df_sub.speed_kmh, color="#2ca02c", lw=1.5, label="Vehicle Speed (km/h)")
    ax2.plot(df_sub.sim_time_s, df_sub.throttle * 50, color="#ff7f0e", linestyle="--", lw=1.2, label="Throttle (scaled x50)")
    ax2.set_ylabel("Speed / Control")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Plot 3: Cumulative Collisions
    ax3.plot(df_sub.sim_time_s, df_sub.collisions, color="#d62728", lw=2, label="Cumulative Collisions (Persistent Pushing)")
    ax3.set_xlabel("Simulation Time (seconds)")
    ax3.set_ylabel("Collisions")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Highlight collision accumulation region
    col_mask = df_sub.collisions > df_sub.collisions.min()
    if col_mask.any():
        first_col_time = df_sub[col_mask].sim_time_s.iloc[0]
        ax3.axvspan(first_col_time, df_sub.sim_time_s.max(), color="#fee", alpha=0.5)
        ax2.axvspan(first_col_time, df_sub.sim_time_s.max(), color="#fee", alpha=0.5)
        ax1.axvspan(first_col_time, df_sub.sim_time_s.max(), color="#fee", alpha=0.5)
        ax3.text(first_col_time + 5, df_sub.collisions.max() * 0.5,
                 "Vehicle stopped by obstacle;\nThrottle stays active (speed < target);\nRepeated collision ticks without braking/redirecting.",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff", ec="#d62728", lw=1.2), fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "figure_15_driving_telemetry_limitation.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def generate_figure_16():
    """Figure 16: Next-Generation Autonomous Architecture: Hybrid Planning & Dynamic Obstacle Avoidance."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")

    # Draw Architecture Flowchart
    # Boxes definition
    boxes = [
        # (x, y, w, h, title, text, color, border)
        (0.04, 0.65, 0.24, 0.28, "1. Perception & Sensing", "• Forward RGB Camera (640x480)\n• YOLOv8 3D Detection\n• Lead Distance & Headway\n• Minimap Waypoint Topology", "#e3f2fd", "#1976d2"),
        (0.36, 0.65, 0.27, 0.28, "2. Next-Gen Path Finder\n(Proposed Upgrade)", "• Curvature-continuous spline\n• Frenet Frame Trajectory Generation\n• Dynamic Obstacle Avoidance\n• Overtake / Lane Change Logic", "#e8f5e9", "#388e3c"),
        (0.71, 0.65, 0.25, 0.28, "3. Lateral Steering Model", "• PilotNet Deep Neural Net\n• Symmetric Data Augmentation\n• Zero-Bias Mirror Inference\n• 0.4 ms/frame Latency", "#ede7f6", "#512da8"),
        
        (0.20, 0.15, 0.28, 0.32, "4. Longitudinal Safety Layer\n(Key Fix Needed)", "• Automatic Emergency Braking (AEB)\n• Time-to-Collision (TTC) monitoring\n• Distance-adaptive throttle/brake\n• Stop & Hold when blocked (<4m)", "#fff3e0", "#f57c00"),
        (0.56, 0.15, 0.35, 0.32, "5. Hybrid Vehicle Controller", "• Blended Steering: 25% Model + 75% Path\n• Coordinated Throttle / Brake / Steer\n• Seamless recovery to clear lanes\n• Telemetry & Incident Logging", "#fce4ec", "#c2185b"),
    ]

    for x, y, w, h, title, text, fc, ec in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top", fontweight="bold", fontsize=10, color=ec, zorder=3)
        ax.text(x + 0.02, y + h - 0.11, text, ha="left", va="top", fontsize=8.2, color="#222", zorder=3, linespacing=1.3)

    # Connecting arrows
    arrow_props = dict(facecolor="#555", edgecolor="#333", width=1.5, headwidth=7, headlength=7, shrink=0.08)
    
    # 1 -> 2
    ax.annotate("", xy=(0.36, 0.79), xytext=(0.28, 0.79), arrowprops=arrow_props)
    # 1 -> 4
    ax.annotate("", xy=(0.28, 0.47), xytext=(0.18, 0.65), arrowprops=arrow_props)
    # 1 -> 3
    ax.annotate("", xy=(0.71, 0.79), xytext=(0.63, 0.79), arrowprops=arrow_props)
    # 2 -> 5
    ax.annotate("", xy=(0.65, 0.47), xytext=(0.50, 0.65), arrowprops=arrow_props)
    # 3 -> 5
    ax.annotate("", xy=(0.78, 0.47), xytext=(0.80, 0.65), arrowprops=arrow_props)
    # 4 -> 5
    ax.annotate("", xy=(0.56, 0.31), xytext=(0.48, 0.31), arrowprops=arrow_props)

    ax.set_title("Autonomous Driving System: Current Progress & Path Finder / Safety Architecture", fontsize=12, pad=12)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "figure_16_nextgen_pathfinder_architecture.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_figure_14()
    generate_figure_15()
    generate_figure_16()
