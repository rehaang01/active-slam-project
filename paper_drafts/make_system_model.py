"""Publication-quality system model figure for the paper.

Three panels on one canvas:
  (a) 3D-perspective multi-altitude warehouse with a detailed quadrotor
      performing a 360-deg LiDAR sweep + dz altitude arrow.
  (b) SLAM back-end — LiDAR returns integrated into a 3D voxel map
      plus pose/covariance; summarised into a 5-channel map tensor
      and a 10-dim scalar vector.
  (c) CoordConv-LSTM actor-critic drawn pictorially with shrinking
      CNN feature maps, scalar MLP, LSTM with recurrent self-loop,
      and separate Actor / Critic heads outputting action & value.
  Bottom legend describes arrow semantics. All graphics composed from
  matplotlib patches (no external icons).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Circle,
                                Ellipse, Rectangle, Polygon, PathPatch)
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent / "figures" / "system_model.png"

# ---- palette ----
C_WAREHOUSE_BG = "#f4f7fb"
C_SLAM_BG      = "#fff8e6"
C_POLICY_BG    = "#eef7ec"
C_SHELF_BROWN  = "#c9a060"
C_SHELF_DARK   = "#5a3a1a"
C_BOX_LIGHT    = "#d4a87a"
C_BOX_MID      = "#bf8a5a"
C_DRONE_BODY   = "#343746"
C_DRONE_ACCENT = "#cc3030"
C_FOV          = "#4fc3f7"
C_SENSE        = "#1565c0"
C_DATA         = "#2e7d32"
C_COMMAND      = "#c62828"
C_RECUR        = "#6a1b9a"


# =============================================================
#                      PRIMITIVES
# =============================================================
def draw_drone(ax, cx, cy, size=1.0, zorder=10):
    """Detailed quadrotor: body + 4 arms + 4 motion-blurred rotors."""
    arm_len = size * 0.50
    arm_pos = [
        (arm_len, arm_len * 0.55),
        (-arm_len, arm_len * 0.55),
        (arm_len, -arm_len * 0.55),
        (-arm_len, -arm_len * 0.55),
    ]
    ax.add_patch(Ellipse((cx, cy - size * 0.45), size * 1.2, size * 0.18,
                         facecolor="black", alpha=0.12, zorder=zorder - 1))
    for dx, dy in arm_pos:
        ax.plot([cx, cx + dx], [cy, cy + dy],
                color="#14151e", linewidth=size * 4.5,
                solid_capstyle="round", zorder=zorder)
    ax.add_patch(Ellipse((cx, cy), size * 0.48, size * 0.28,
                         facecolor=C_DRONE_BODY, edgecolor="#0b0d14",
                         linewidth=1.0, zorder=zorder + 1))
    ax.add_patch(Ellipse((cx - size * 0.08, cy + size * 0.05),
                         size * 0.20, size * 0.08,
                         facecolor="#a0a5b8", edgecolor="none",
                         alpha=0.55, zorder=zorder + 2))
    ax.add_patch(Rectangle((cx - size * 0.035, cy - size * 0.16),
                           size * 0.07, size * 0.09,
                           facecolor=C_DRONE_ACCENT, edgecolor="#661414",
                           linewidth=0.3, zorder=zorder + 2))
    for dx, dy in arm_pos:
        rx, ry = cx + dx, cy + dy
        ax.add_patch(Circle((rx, ry), size * 0.09,
                            facecolor="#0f1018", edgecolor="none",
                            zorder=zorder + 2))
        ax.add_patch(Circle((rx, ry), size * 0.26,
                            facecolor="#8a8fa3", edgecolor="#3a3e50",
                            linewidth=0.6, alpha=0.4, zorder=zorder + 1))
        for angle in (0, 60, 120):
            a = np.radians(angle)
            ax.plot([rx - np.cos(a) * size * 0.24, rx + np.cos(a) * size * 0.24],
                    [ry - np.sin(a) * size * 0.24, ry + np.sin(a) * size * 0.24],
                    color="#cfd3e0", linewidth=0.4, alpha=0.55,
                    zorder=zorder + 1)


def draw_shelf3d(ax, x, y, w=1.0, h=1.15, depth=0.18):
    ax.add_patch(Polygon(
        [(x + w, y), (x + w + depth, y + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        facecolor="#755a35", edgecolor="#2a1a08", linewidth=0.4, zorder=3))
    ax.add_patch(Polygon(
        [(x, y + h), (x + w, y + h),
         (x + w + depth, y + h + depth), (x + depth, y + h + depth)],
        facecolor="#a28155", edgecolor="#2a1a08", linewidth=0.4, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=C_SHELF_BROWN,
                           edgecolor=C_SHELF_DARK, linewidth=0.8, zorder=4))
    ax.add_patch(Rectangle((x + 0.03, y), 0.08, h,
                           facecolor=C_SHELF_DARK, zorder=5))
    ax.add_patch(Rectangle((x + w - 0.11, y), 0.08, h,
                           facecolor=C_SHELF_DARK, zorder=5))
    for sy in [y + h * 0.33, y + h * 0.66, y + h - 0.04]:
        ax.add_patch(Rectangle((x + 0.02, sy), w - 0.04, 0.05,
                               facecolor="#4a3620",
                               edgecolor="#2a1a08", linewidth=0.3, zorder=6))
    rng = np.random.default_rng(int(x * 100 + y * 17))
    for sy in [y + h * 0.66 + 0.05, y + h * 0.33 + 0.05]:
        off = 0.08
        while off < w - 0.22:
            bw = min(rng.uniform(0.17, 0.30), w - off - 0.05)
            c = rng.choice([C_BOX_LIGHT, C_BOX_MID])
            ax.add_patch(Rectangle((x + off, sy), bw, 0.20,
                                   facecolor=c, edgecolor="#5a3a1a",
                                   linewidth=0.35, zorder=7))
            ax.plot([x + off + bw / 2] * 2, [sy, sy + 0.20],
                    color="#5a3a1a", linewidth=0.3, zorder=8)
            off += bw + 0.06


def nn_box(ax, x, y, w, h, text, fill, edge, fontsize=12, zorder=25,
           fontweight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=zorder))
    ax.text(x + w / 2, y, text, ha="center", va="center",
            fontsize=fontsize, color="#0c1117", zorder=zorder + 1,
            fontweight=fontweight)


def panel_bg(ax, x, y, w, h, label, colour, zorder=0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=colour, edgecolor="#7a8595",
        linewidth=1.2, zorder=zorder))
    ax.text(x + 0.25, y + h - 0.38, label,
            fontsize=15, fontweight="bold", style="italic",
            color="#2e3b4d", zorder=zorder + 1)


# =============================================================
#                          COMPOSE
# =============================================================
def main() -> None:
    fig = plt.figure(figsize=(22, 11.2), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.3, 11.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Panel backgrounds ---
    panel_bg(ax,  0.2, 1.1, 11.5, 9.6,  "(a) Multi-altitude warehouse", C_WAREHOUSE_BG)
    panel_bg(ax, 12.0, 1.1,  6.3, 9.6,  "(b) SLAM back-end",            C_SLAM_BG)
    panel_bg(ax, 18.5, 1.1, 11.3, 9.6,  "(c) RL policy — CoordConv--LSTM Actor-Critic",
             C_POLICY_BG)

    # ==========================================================
    # PANEL (a) — WAREHOUSE
    # ==========================================================
    layer_y0 = [1.85, 3.45, 5.05, 6.65]
    layer_h  = 1.35
    layer_lbl = ["1\\,m", "2\\,m", "3\\,m", "4\\,m"]
    layer_shade = ["#ffffff", "#fbfdff", "#f3f8ff", "#e8f1fb"]
    skew = 0.35

    for y, lbl, sh in zip(layer_y0, layer_lbl, layer_shade):
        ax.add_patch(Polygon(
            [(0.9, y), (10.6, y), (10.6 + skew, y + layer_h),
             (0.9 + skew, y + layer_h)],
            facecolor=sh, edgecolor="#3a4a5c", linewidth=1.3, zorder=2))
        ax.text(0.5, y + layer_h / 2, f"${lbl}$",
                ha="right", va="center", fontsize=16, fontweight="bold",
                color="#1a2a3c")

    shelves = {
        0: [1.6, 4.2, 6.8, 9.1],
        1: [2.0, 5.2, 8.5],
        2: [6.2, 8.8],
        3: [1.5, 4.2, 6.6, 8.9],
    }
    for lid, xs in shelves.items():
        y_base = layer_y0[lid] + 0.05
        for sx in xs:
            draw_shelf3d(ax, sx, y_base, w=1.05, h=1.20)

    # UAV
    uav_cx = 3.3
    uav_cy = layer_y0[2] + layer_h / 2 + 0.05
    draw_drone(ax, uav_cx, uav_cy, size=1.25)
    # UAV label sits in an opaque badge so it doesn't overlap layer edges
    ax.text(uav_cx + 0.88, uav_cy - 0.02, "UAV", ha="left", va="center",
            fontsize=14, fontweight="bold", color="#0c1117", zorder=30,
            bbox=dict(facecolor="white", edgecolor="#0c1117",
                      boxstyle="round,pad=0.22", alpha=0.92))

    # LiDAR rays 360 deg
    n_rays = 36
    max_r = 2.0
    for i in range(n_rays):
        theta = 2 * np.pi * i / n_rays
        dx = np.cos(theta) * max_r
        dy = np.sin(theta) * max_r * 0.55
        ax.plot([uav_cx, uav_cx + dx], [uav_cy, uav_cy + dy],
                color=C_FOV, linewidth=0.55, alpha=0.55, zorder=9)
    # LiDAR label placed clear of the warehouse, in empty air between
    # the warehouse and the SLAM panel
    ax.annotate("360$^\\circ$ LiDAR\n(36 beams)",
                xy=(uav_cx + 1.6, uav_cy - 0.15),
                xytext=(10.9, 2.5),
                fontsize=13, color="#01579b", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#01579b",
                                linewidth=1.4),
                zorder=31,
                bbox=dict(facecolor="white", edgecolor="#01579b",
                          boxstyle="round,pad=0.25", alpha=0.95))

    # dz double arrow
    dz_x = uav_cx - 1.05
    ax.annotate("", xy=(dz_x, layer_y0[-1] + layer_h - 0.10),
                xytext=(dz_x, layer_y0[0] + 0.15),
                arrowprops=dict(arrowstyle="<->", color=C_COMMAND,
                                linewidth=2.8, mutation_scale=22),
                zorder=9)
    ax.text(dz_x - 0.48, uav_cy + 0.35, "$dz$", fontsize=18,
            color=C_COMMAND, fontweight="bold", style="italic")
    ax.text(dz_x - 0.58, uav_cy - 0.55, "altitude\nvote",
            fontsize=11, color=C_COMMAND, ha="center")

    # Axes indicator
    ox, oy = 9.9, 1.45
    ax.annotate("", xy=(ox + 0.7, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color="#333", linewidth=1.3))
    ax.annotate("", xy=(ox, oy + 0.7), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color="#333", linewidth=1.3))
    ax.annotate("", xy=(ox + 0.45, oy + 0.45), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color="#333", linewidth=1.3))
    ax.text(ox + 0.82, oy, "$x$", fontsize=13)
    ax.text(ox - 0.08, oy + 0.85, "$z$", fontsize=13)
    ax.text(ox + 0.55, oy + 0.58, "$y$", fontsize=13)

    # ==========================================================
    # PANEL (b) — SLAM back-end
    # ==========================================================
    # LiDAR returns arrow crossing from panel (a)
    ax.annotate("",
                xy=(12.25, 8.4),
                xytext=(10.65, 7.4),
                arrowprops=dict(arrowstyle="->", color=C_SENSE,
                                linewidth=2.4, mutation_scale=19),
                zorder=25)
    ax.text(11.35, 8.15, "LiDAR\nreturns", fontsize=13, color=C_SENSE,
            fontweight="bold", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.88))

    # Three SLAM subcomponents
    nn_box(ax, 12.35, 8.35, 5.6, 1.1,
           "3D Occupancy Map  $\\mathcal{M}_t$",
           "#d4e7f6", "#0b3d91", fontsize=14, fontweight="bold")
    vx0, vy0 = 16.9, 7.95
    for ix in range(3):
        for iy in range(2):
            shade = "#7fb1e0" if (ix + iy) % 2 == 0 else "#3c7cb8"
            ax.add_patch(Rectangle((vx0 + ix * 0.25, vy0 + iy * 0.25),
                                   0.22, 0.22, facecolor=shade,
                                   edgecolor="#0b3d91", linewidth=0.6,
                                   zorder=27))

    nn_box(ax, 12.35, 6.60, 5.6, 1.1,
           "Pose Estimate  $(\\hat{x}_t, \\hat{y}_t, \\hat{z}_t)$",
           "#d4e7f6", "#0b3d91", fontsize=14, fontweight="bold")

    nn_box(ax, 12.35, 4.85, 5.6, 1.1,
           "Uncertainty surrogate  $\\sigma_t \\!\\in\\! [0,4]$",
           "#d4e7f6", "#0b3d91", fontsize=14, fontweight="bold")
    ax.add_patch(Ellipse((17.0, 4.85), 0.60, 0.32,
                         facecolor="none", edgecolor="#0b3d91",
                         linewidth=1.8, zorder=27))
    ax.add_patch(Circle((17.0, 4.85), 0.06,
                        facecolor="#0b3d91", zorder=28))

    # Summary arrow
    ax.annotate("", xy=(15.15, 3.50), xytext=(15.15, 4.25),
                arrowprops=dict(arrowstyle="->", color="#555",
                                linewidth=2.0, mutation_scale=16),
                zorder=26)
    ax.text(15.15, 3.88, "summarise", fontsize=11.5, color="#555",
            ha="center", style="italic",
            bbox=dict(facecolor=C_SLAM_BG, edgecolor="none", pad=1))

    # Observation bundle: map tensor + scalar vector (moved UP out of the loop)
    # 5-channel map tensor
    mt_x, mt_y = 12.5, 3.00
    for i in range(5):
        shade = ["#64b5f6", "#4fc3f7", "#29b6f6", "#03a9f4", "#0288d1"][i]
        ax.add_patch(FancyBboxPatch(
            (mt_x + i * 0.07, mt_y - i * 0.07), 1.05, 0.95,
            boxstyle="round,pad=0.005,rounding_size=0.04",
            facecolor=shade, edgecolor="#01579b",
            linewidth=0.6, zorder=30 - i))
    ax.text(mt_x + 0.85, mt_y - 0.45,
            "Map Tensor\n$M_t\\in\\mathbb{R}^{5\\times48\\times48}$",
            fontsize=12, ha="center", color="#01579b", fontweight="bold")

    # 10-d scalar vector
    sv_x, sv_y = 14.95, 2.95
    for i in range(10):
        c = plt.cm.Oranges(0.3 + 0.06 * i)
        ax.add_patch(Rectangle((sv_x, sv_y + i * 0.10), 0.55, 0.095,
                               facecolor=c, edgecolor="#b15000",
                               linewidth=0.4, zorder=30))
    ax.text(sv_x + 0.28, sv_y - 0.32,
            "Scalars\n$v_t\\in[0,1]^{10}$",
            fontsize=12, ha="center", color="#b15000", fontweight="bold")

    # Observation -> policy arrow (moved up to match new bundle y)
    ax.annotate("", xy=(18.65, 3.30), xytext=(16.5, 3.30),
                arrowprops=dict(arrowstyle="->", color=C_DATA,
                                linewidth=2.6, mutation_scale=19),
                zorder=35)
    ax.text(17.6, 3.65, "observation $o_t$", fontsize=13.5,
            color=C_DATA, fontweight="bold", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.92))

    # ==========================================================
    # PANEL (c) — Policy
    # ==========================================================
    ax.text(24.1, 9.70, "Actor-Critic with Shared Backbone",
            fontsize=14, color="#0d1f0b",
            ha="center", style="italic")

    # CNN stack representation (shrunk + moved left to give room at right)
    cnn_specs = [
        (19.55, 7.05, 1.25, 3, "#8dc9a0"),   # stage 1
        (21.10, 7.05, 0.95, 5, "#66b486"),   # stage 2
        (22.45, 7.05, 0.68, 7, "#3fa06b"),   # stage 3
    ]
    for (xc, yc, sz, n, col) in cnn_specs:
        off = 0.07
        for j in range(n):
            ax.add_patch(FancyBboxPatch(
                (xc - sz / 2 + off * (n - 1 - j),
                 yc - sz / 2 + off * (n - 1 - j)),
                sz, sz,
                boxstyle="round,pad=0.005,rounding_size=0.03",
                facecolor=col, edgecolor="#14532d",
                linewidth=0.4, alpha=1 - 0.04 * j, zorder=21 - j))
    stage_txt = ["CoordConv2D\n$c{=}32$\nstride 2",
                 "Conv2D\n$c{=}64$\nstride 2",
                 "Conv2D\n$c{=}64$\nstride 2"]
    for (xc, _, sz, _, _), t in zip(cnn_specs, stage_txt):
        ax.text(xc, 7.05 - sz / 2 - 0.55, t,
                fontsize=11, ha="center", color="#14532d")

    prev_x = None
    for (xc, yc, sz, _, _) in cnn_specs:
        if prev_x is not None:
            ax.annotate("", xy=(xc - sz / 2, yc),
                        xytext=(prev_x, yc),
                        arrowprops=dict(arrowstyle="->", color="#14532d",
                                        linewidth=1.5, mutation_scale=12),
                        zorder=25)
        prev_x = xc + sz / 2 + 0.04

    # Conv2D + AvgPool6 + Flatten -> R^2304
    FL_W, FL_H = 1.55, 1.15
    FL_CX, FL_CY = 23.60, 7.05
    nn_box(ax, FL_CX, FL_CY, FL_W, FL_H,
           "Conv2D $c{=}64$\nAvgPool$_6$\n+ Flatten",
           "#c8e6c9", "#2e7d32", fontsize=11)
    ax.annotate("", xy=(FL_CX - FL_W/2, FL_CY), xytext=(prev_x, FL_CY),
                arrowprops=dict(arrowstyle="->", color="#14532d",
                                linewidth=1.5, mutation_scale=12), zorder=25)

    # Scalar stream
    sc_x, sc_y = 19.85, 4.25
    for i in range(10):
        ax.add_patch(Rectangle((sc_x, sc_y + i * 0.075), 0.38, 0.068,
                               facecolor=plt.cm.Oranges(0.3 + 0.06 * i),
                               edgecolor="#b15000", linewidth=0.3,
                               zorder=21))
    ax.text(sc_x + 0.19, sc_y - 0.32, "Scalars\n$v_t$",
            fontsize=11, ha="center", color="#b15000", fontweight="bold")
    FC64_W, FC64_H = 1.55, 0.85
    FC64_CX, FC64_CY = 21.30, 4.70
    nn_box(ax, FC64_CX, FC64_CY, FC64_W, FC64_H,
           "LayerNorm\nFC-64$\\to$FC-32",
           "#ffe0b2", "#b15000", fontsize=11)
    ax.annotate("", xy=(FC64_CX - FC64_W/2, FC64_CY),
                xytext=(sc_x + 0.38, FC64_CY),
                arrowprops=dict(arrowstyle="->", color="#b15000",
                                linewidth=1.3, mutation_scale=11), zorder=25)

    # Concat
    CC_W, CC_H = 1.35, 0.95
    CC_CX, CC_CY = 24.75, 5.90
    nn_box(ax, CC_CX, CC_CY, CC_W, CC_H,
           "Concat+MLP\n$\\mathbb{R}^{256}$",
           "#e1bee7", "#6a1b9a", fontsize=12)
    # Flatten -> Concat (from FC-256 bottom to Concat top)
    ax.annotate("", xy=(CC_CX, CC_CY + CC_H/2),
                xytext=(FL_CX, FL_CY - FL_H/2),
                arrowprops=dict(arrowstyle="->", color="#14532d",
                                linewidth=1.5, mutation_scale=12,
                                connectionstyle="arc3,rad=0.15"),
                zorder=25)
    # FC-64 -> Concat (from FC-64 right to Concat bottom-left)
    ax.annotate("", xy=(CC_CX, CC_CY - CC_H/2),
                xytext=(FC64_CX + FC64_W/2, FC64_CY),
                arrowprops=dict(arrowstyle="->", color="#b15000",
                                linewidth=1.4, mutation_scale=12,
                                connectionstyle="arc3,rad=-0.15"),
                zorder=25)

    # LSTM
    LS_W, LS_H = 1.35, 0.95
    LS_CX, LS_CY = 26.35, 5.90
    nn_box(ax, LS_CX, LS_CY, LS_W, LS_H, "LSTM\n$h_t\\in\\mathbb{R}^{128}$",
           "#b2dfdb", "#00695c", fontsize=12)
    # Concat -> LSTM
    ax.annotate("", xy=(LS_CX - LS_W/2, LS_CY),
                xytext=(CC_CX + CC_W/2, CC_CY),
                arrowprops=dict(arrowstyle="->", color="#444",
                                linewidth=1.6, mutation_scale=12), zorder=25)

    # Recurrence loop on top of LSTM
    ax.annotate("",
                xy=(LS_CX - LS_W/2 + 0.15, LS_CY + LS_H/2 + 0.02),
                xytext=(LS_CX + LS_W/2 - 0.15, LS_CY + LS_H/2 + 0.02),
                arrowprops=dict(arrowstyle="->", color=C_RECUR,
                                linewidth=2.0, mutation_scale=14,
                                connectionstyle="arc3,rad=-0.7",
                                linestyle="dashed"),
                zorder=26)
    ax.text(LS_CX, LS_CY + 1.15, "$h_{t-1}$", fontsize=15, color=C_RECUR,
            ha="center", fontweight="bold", style="italic")

    # Actor head + Critic head (well-separated from LSTM)
    AC_W, AC_H = 1.30, 0.90
    AC_CX = 28.35                                # was 28.75, shifted left
    AC_CY_ACTOR = 7.00
    AC_CY_CRITIC = 4.80
    nn_box(ax, AC_CX, AC_CY_ACTOR, AC_W, AC_H, "Actor\n$\\pi(a_t|o_t)$",
           "#ffccbc", "#bf360c", fontsize=11.5, fontweight="bold")
    nn_box(ax, AC_CX, AC_CY_CRITIC, AC_W, AC_H, "Critic\n$V(o_t)$",
           "#ffccbc", "#bf360c", fontsize=11.5, fontweight="bold")
    # LSTM -> Actor (arrow from LSTM top-right into Actor bottom-left)
    ax.annotate("",
                xy=(AC_CX - AC_W/2, AC_CY_ACTOR - AC_H/2 + 0.05),
                xytext=(LS_CX + LS_W/2, LS_CY + LS_H/2 - 0.15),
                arrowprops=dict(arrowstyle="->", color="#bf360c",
                                linewidth=1.6, mutation_scale=12),
                zorder=26)
    # LSTM -> Critic (arrow from LSTM bottom-right into Critic top-left)
    ax.annotate("",
                xy=(AC_CX - AC_W/2, AC_CY_CRITIC + AC_H/2 - 0.05),
                xytext=(LS_CX + LS_W/2, LS_CY - LS_H/2 + 0.15),
                arrowprops=dict(arrowstyle="->", color="#bf360c",
                                linewidth=1.6, mutation_scale=12),
                zorder=26)

    # Action box
    ACT_CX, ACT_CY = 25.40, 2.65
    ACT_W, ACT_H = 4.00, 1.00
    nn_box(ax, ACT_CX, ACT_CY, ACT_W, ACT_H,
           "Action  $a_t = (dx, dy, dz) \\in [-1, 1]^3$",
           "#ffe0b2", C_COMMAND, fontsize=13.5, fontweight="bold")
    # Actor -> Action (right side path — down along right edge of panel)
    action_path = MplPath(
        [(AC_CX + AC_W/2, AC_CY_ACTOR),
         (29.55, AC_CY_ACTOR),
         (29.55, ACT_CY),
         (ACT_CX + ACT_W/2 + 0.05, ACT_CY)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(PathPatch(action_path, facecolor="none",
                           edgecolor=C_COMMAND, linewidth=2.2, zorder=26))
    ax.annotate("", xy=(ACT_CX + ACT_W/2, ACT_CY),
                xytext=(ACT_CX + ACT_W/2 + 0.12, ACT_CY),
                arrowprops=dict(arrowstyle="->", color=C_COMMAND,
                                linewidth=2.2, mutation_scale=16), zorder=27)

    # ==========================================================
    # COMMAND LOOP back to UAV
    # Route at y = 0.3 so it does NOT cross any panel content
    # ==========================================================
    loop_y = 0.3
    loop = MplPath(
        [(ACT_CX - ACT_W/2, ACT_CY - ACT_H/2),   # start at bottom-left corner
         (ACT_CX - ACT_W/2, loop_y),
         (uav_cx, loop_y),
         (uav_cx, layer_y0[2] - 0.10)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(PathPatch(loop, facecolor="none",
                           edgecolor=C_COMMAND, linewidth=2.2, zorder=14))
    ax.annotate("", xy=(uav_cx, layer_y0[2] - 0.10),
                xytext=(uav_cx, layer_y0[2] - 0.25),
                arrowprops=dict(arrowstyle="->", color=C_COMMAND,
                                linewidth=2.2, mutation_scale=18), zorder=15)
    ax.text(16.0, loop_y + 0.18,
            "command: drive flight controller",
            fontsize=13, color=C_COMMAND, fontweight="bold", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.92))

    # ==========================================================
    # LEGEND
    # ==========================================================
    lg_y = -0.85
    entries = [
        (C_SENSE,   "sensor / perception",                    "solid"),
        (C_DATA,    "observation / data flow",                "solid"),
        (C_COMMAND, "control command / action",               "solid"),
        (C_RECUR,   "temporal recurrence ($h_{t-1}$)",       "dashed"),
    ]
    xs = 1.5
    for col, lbl, ls in entries:
        ax.annotate("", xy=(xs + 1.1, lg_y), xytext=(xs, lg_y),
                    arrowprops=dict(arrowstyle="->", color=col,
                                    linewidth=2.4, mutation_scale=17,
                                    linestyle=ls))
        ax.text(xs + 1.2, lg_y, lbl, fontsize=13, va="center",
                color="#0c1117")
        xs += 6.2

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
