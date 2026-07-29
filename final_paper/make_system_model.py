"""Render the system-model figure for the paper.

Draws a four-altitude warehouse with shelves and a quadrotor UAV on the left;
the perception-action pipeline (Observation -> Policy -> Action -> SLAM) on the
right; and clean non-overlapping connecting arrows. No external icon downloads;
all graphics are composed from matplotlib patches.

Output: paper_drafts/figures/system_model.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Circle,
                                Ellipse, Rectangle, Polygon)
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


OUT = Path(__file__).resolve().parent / "figures" / "system_model.png"


# -------------------- drawing primitives --------------------
def draw_shelf(ax, x, y, width=1.05, total_h=0.9):
    """A warehouse pallet-rack with two shelves of boxes and side uprights.
    Drawn with a mild 3D offset (right-side shadow) for depth."""
    # Right-side shadow face
    shadow = Polygon(
        [(x + width, y),
         (x + width + 0.08, y + 0.08),
         (x + width + 0.08, y + total_h + 0.08),
         (x + width, y + total_h)],
        facecolor="#6b5535", edgecolor="#3a2a10", linewidth=0.4, zorder=3,
    )
    ax.add_patch(shadow)
    # Top face
    top = Polygon(
        [(x, y + total_h),
         (x + width, y + total_h),
         (x + width + 0.08, y + total_h + 0.08),
         (x + 0.08, y + total_h + 0.08)],
        facecolor="#967845", edgecolor="#3a2a10", linewidth=0.4, zorder=3,
    )
    ax.add_patch(top)

    # Front frame (back wall of shelf)
    ax.add_patch(Rectangle((x, y), width, total_h, facecolor="#c9a060",
                           edgecolor="#3a2a10", linewidth=0.7, zorder=4))
    # Vertical uprights (dark)
    ax.add_patch(Rectangle((x + 0.02, y), 0.08, total_h,
                           facecolor="#3a2a10", edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x + width - 0.10, y), 0.08, total_h,
                           facecolor="#3a2a10", edgecolor="none", zorder=5))
    # Horizontal shelves (planks)
    for shelf_y in [y + total_h * 0.33, y + total_h * 0.66, y + total_h - 0.04]:
        ax.add_patch(Rectangle((x + 0.02, shelf_y), width - 0.04, 0.045,
                               facecolor="#4a3620", edgecolor="#2a1a08",
                               linewidth=0.3, zorder=6))
    # Boxes on the two upper shelves
    box_colours = ["#d4a87a", "#bf8a5a", "#cfa070", "#b88050"]
    # Upper row
    for i, bx_off in enumerate([0.14, 0.42, 0.70]):
        bw, bh = 0.22, 0.18
        ax.add_patch(Rectangle((x + bx_off, y + total_h * 0.66 + 0.045),
                               bw, bh, facecolor=box_colours[i % 4],
                               edgecolor="#5a3a1a", linewidth=0.4, zorder=7))
        ax.plot([x + bx_off + bw / 2, x + bx_off + bw / 2],
                [y + total_h * 0.66 + 0.045, y + total_h * 0.66 + 0.045 + bh],
                color="#5a3a1a", linewidth=0.35, zorder=8)
    # Middle row
    for i, bx_off in enumerate([0.18, 0.55]):
        bw, bh = 0.30, 0.22
        ax.add_patch(Rectangle((x + bx_off, y + total_h * 0.33 + 0.045),
                               bw, bh, facecolor=box_colours[(i + 2) % 4],
                               edgecolor="#5a3a1a", linewidth=0.4, zorder=7))


def draw_drone(ax, cx, cy, size=1.0):
    """Composed quadrotor: 4 arms + 4 rotor discs with motion-blur + body."""
    arm_len = size * 0.48
    arm_pos = [
        (arm_len, arm_len * 0.55),
        (-arm_len, arm_len * 0.55),
        (arm_len, -arm_len * 0.55),
        (-arm_len, -arm_len * 0.55),
    ]

    # Arms (drawn first, under rotors)
    for dx, dy in arm_pos:
        ax.plot([cx, cx + dx], [cy, cy + dy],
                color="#1a1a2a", linewidth=3.3, solid_capstyle="round",
                zorder=10)

    # Central body (ellipse with highlight for gloss)
    ax.add_patch(Ellipse((cx, cy), size * 0.44, size * 0.26,
                         facecolor="#343746", edgecolor="#0f111a",
                         linewidth=0.9, zorder=11))
    ax.add_patch(Ellipse((cx - size * 0.06, cy + size * 0.04),
                         size * 0.16, size * 0.07,
                         facecolor="#8a8fa3", edgecolor="none",
                         alpha=0.55, zorder=12))
    # Central small antenna / camera bump
    ax.add_patch(Rectangle((cx - size * 0.025, cy - size * 0.14),
                           size * 0.05, size * 0.08,
                           facecolor="#cc3030", edgecolor="#661414",
                           linewidth=0.25, zorder=12))

    # Rotors (motor base + spinning disc + motion blur lines)
    for dx, dy in arm_pos:
        rx, ry = cx + dx, cy + dy
        # Motor hub
        ax.add_patch(Circle((rx, ry), size * 0.075,
                            facecolor="#141520", edgecolor="none", zorder=12))
        # Semi-transparent disc
        ax.add_patch(Circle((rx, ry), size * 0.22,
                            facecolor="#7a7e90", edgecolor="#3a3e50",
                            linewidth=0.5, alpha=0.42, zorder=11))
        # Motion-blur lines
        for angle in (0, 60, 120):
            a = np.radians(angle)
            ax.plot([rx - np.cos(a) * size * 0.2, rx + np.cos(a) * size * 0.2],
                    [ry - np.sin(a) * size * 0.2, ry + np.sin(a) * size * 0.2],
                    color="#bcc0cf", linewidth=0.35, alpha=0.55, zorder=11)
        # Tiny rotor cap highlight
        ax.add_patch(Circle((rx, ry), size * 0.04,
                            facecolor="#5a5d70", edgecolor="none", zorder=13))


def pipe_box(ax, x, y, w, h, text, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.09",
                                facecolor=fill, edgecolor=edge,
                                linewidth=1.4, zorder=20))
    ax.text(x + w / 2, y, text, ha="center", va="center",
            fontsize=10.5, color="#0c1117", zorder=21)


# -------------------- compose figure --------------------
def main() -> None:
    fig = plt.figure(figsize=(14, 6.8), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.2, 7.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Warehouse panel background
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 10.8, 6.45,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#f6f8fb", edgecolor="#bac4d0",
                                linewidth=1.0, zorder=1))

    # Altitude layers
    layer_ys = [0.45, 1.75, 3.05, 4.35]    # bottoms, 1m..4m
    layer_h = 1.25
    layer_labels = ["1 m", "2 m", "3 m", "4 m"]
    layer_fills = ["#ffffff", "#fafcff", "#f3f7fd", "#eaf1f9"]

    for y, label, fc in zip(layer_ys, layer_labels, layer_fills):
        ax.add_patch(Rectangle((0.65, y), 9.9, layer_h,
                               facecolor=fc, edgecolor="#3a4a5c",
                               linewidth=1.1, zorder=2))
        ax.text(0.5, y + layer_h / 2, label, ha="right", va="center",
                fontsize=10.5, fontweight="bold", color="#2a3a4c")

    ax.text(0.65, 6.25, "Multi-altitude warehouse",
            fontsize=11, style="italic", color="#3a4a5c")
    ax.text(0.4, 5.85, "altitude\nlayers", fontsize=8.5,
            style="italic", color="#4a5a6c", ha="left")

    # Shelves on each altitude. UAV sits on 3m (layer_ys[2]) leaving a gap.
    # Kept sparse so arrow routing is unambiguous.
    shelf_positions = {
        0: [1.6, 4.2, 6.8, 9.1],             # 1m (leave lane 2.8-4.0 for command arrow)
        1: [2.1, 4.5, 7.8],                   # 2m
        2: [6.6, 8.6],                        # 3m (open space around UAV)
        3: [1.4, 4.0, 6.2, 8.4],             # 4m
    }
    for lid, xs in shelf_positions.items():
        y_base = layer_ys[lid] + 0.05
        for x in xs:
            draw_shelf(ax, x, y_base, width=0.95, total_h=1.05)

    # UAV on the 3m layer
    uav_cx = 3.3
    uav_cy = layer_ys[2] + layer_h / 2
    draw_drone(ax, uav_cx, uav_cy, size=1.25)
    ax.text(uav_cx, uav_cy - 0.58, "UAV", ha="center", fontsize=10,
            fontweight="bold", color="#1a1a2a", zorder=20)

    # FOV cone (to the right)
    fov = Polygon(
        [(uav_cx + 0.35, uav_cy + 0.04),
         (uav_cx + 2.9, uav_cy + 0.55),
         (uav_cx + 2.9, uav_cy - 0.55)],
        facecolor="#4fc3f7", edgecolor="#0288d1", alpha=0.28,
        linewidth=0.6, zorder=7,
    )
    ax.add_patch(fov)
    ax.text(uav_cx + 2.0, uav_cy + 0.32, "FOV", fontsize=9,
            color="#01579b", style="italic", fontweight="bold", zorder=21)

    # dz arrow (altitude change), placed to the left of the UAV
    dz_x = uav_cx - 0.95
    ax.annotate("", xy=(dz_x, layer_ys[-1] + layer_h - 0.18),
                xytext=(dz_x, layer_ys[0] + 0.18),
                arrowprops=dict(arrowstyle="<->", color="#c62828",
                                linewidth=2.0, mutation_scale=20),
                zorder=9)
    ax.text(dz_x - 0.32, uav_cy, r"$dz$", fontsize=13, color="#c62828",
            fontweight="bold", style="italic", zorder=21)

    # ---------- Pipeline on the right ----------
    box_w, box_h = 3.7, 0.95
    box_x = 11.4

    pipe_y = {
        "obs":    5.55,
        "policy": 4.20,
        "action": 2.85,
        "slam":   1.50,
    }

    pipe_box(ax, box_x, pipe_y["obs"], box_w, box_h,
             "Observation\n$o_t = (M_t,\\,s_t)$",
             fill="#a5d6a7", edge="#2e7d32")
    pipe_box(ax, box_x, pipe_y["policy"], box_w, box_h,
             "RL Policy\n$\\pi(a_t \\,|\\, o_t)$",
             fill="#ffcc80", edge="#ef6c00")
    pipe_box(ax, box_x, pipe_y["action"], box_w, box_h,
             "Action $(dx,\\, dy,\\, dz)$",
             fill="#ffcc80", edge="#ef6c00")
    pipe_box(ax, box_x, pipe_y["slam"], box_w, box_h,
             "SLAM Back-end",
             fill="#a5d6a7", edge="#2e7d32")

    # Vertical arrows between pipeline boxes (clean, down the centre)
    centre_x = box_x + box_w / 2
    for (y_top, y_bot) in [
        (pipe_y["obs"]    - box_h / 2, pipe_y["policy"] + box_h / 2),
        (pipe_y["policy"] - box_h / 2, pipe_y["action"] + box_h / 2),
        (pipe_y["action"] - box_h / 2, pipe_y["slam"]   + box_h / 2),
    ]:
        ax.annotate("", xy=(centre_x, y_bot), xytext=(centre_x, y_top),
                    arrowprops=dict(arrowstyle="->", color="#263238",
                                    linewidth=1.6, mutation_scale=15),
                    zorder=22)

    # ---------- connecting arrows (non-overlapping) ----------
    # "sense" arrow : warehouse (top-right) -> Observation box (left edge)
    ax.annotate("",
                xy=(box_x - 0.08, pipe_y["obs"]),
                xytext=(10.95, pipe_y["obs"]),
                arrowprops=dict(arrowstyle="->", color="#0b3d91",
                                linewidth=1.8, mutation_scale=16),
                zorder=22)
    ax.text((10.95 + box_x - 0.08) / 2, pipe_y["obs"] - 0.26, "sense",
            fontsize=9.5, color="#0b3d91", ha="center", va="top",
            style="italic", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9))

    # "command" arrow : Action box -> UAV
    # Route it in the narrow gap BETWEEN 2m and 3m layers so it never crosses
    # shelves, FOV, or the dz arrow.
    cmd_route_y = layer_ys[2] - 0.20     # gap between 2m and 3m layers

    command_path = MplPath(
        [
            (box_x, pipe_y["action"]),                       # start: left edge of Action
            (box_x - 0.45, pipe_y["action"]),
            (box_x - 0.45, cmd_route_y),                     # down to gap lane
            (uav_cx + 0.05, cmd_route_y),                    # left toward UAV
            (uav_cx + 0.05, uav_cy - 0.38),                  # up to UAV (below body)
        ],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
         MplPath.LINETO, MplPath.LINETO]
    )
    ax.add_patch(PathPatch(command_path, facecolor="none",
                           edgecolor="#b55a00", linewidth=2.0, zorder=22))
    # Arrow head at the UAV end
    ax.annotate("", xy=(uav_cx + 0.05, uav_cy - 0.25),
                xytext=(uav_cx + 0.05, uav_cy - 0.36),
                arrowprops=dict(arrowstyle="->", color="#b55a00",
                                linewidth=2.0, mutation_scale=17),
                zorder=23)
    ax.text(8.0, cmd_route_y + 0.16, "command",
            fontsize=9.5, color="#b55a00", ha="center", style="italic",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.92))

    # "update" arrow : SLAM back-end -> Observation, looping right of the pipeline
    upd_x = box_x + box_w + 0.45
    ax.annotate("", xy=(box_x + box_w + 0.05, pipe_y["obs"]),
                xytext=(box_x + box_w + 0.05, pipe_y["slam"]),
                arrowprops=dict(arrowstyle="->", color="#1b5e20",
                                linewidth=1.6, mutation_scale=14,
                                connectionstyle="arc3,rad=0.35"),
                zorder=22)
    ax.text(upd_x + 0.35, (pipe_y["obs"] + pipe_y["slam"]) / 2,
            "update", fontsize=9, color="#1b5e20",
            ha="center", va="center", style="italic", rotation=90)

    # Legend for arrow colours (small, bottom-left of pipeline)
    lg_x = 0.25
    lg_y = 6.65
    # (keeping it minimal — labels are in-place already; no separate legend)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
