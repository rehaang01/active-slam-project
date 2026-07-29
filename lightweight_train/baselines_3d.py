"""Baselines for 3D altitude-aware Active SLAM."""
import os
import sys

import numpy as np
import math

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from transfer_config import N_ALTITUDES


class RandomWalk3D:
    name = "Random Walk"
    def reset(self): pass
    def get_action(self, obs, info):
        return np.random.uniform(-1, 1, 3).astype(np.float32)


class NearestFrontier3D:
    """Go to nearest frontier at current alt. Switch alt when coverage plateaus."""
    name = "Nearest Frontier"
    def __init__(self): self.stuck_steps = 0; self.prev_cov = 0.0
    def reset(self): self.stuck_steps = 0; self.prev_cov = 0.0

    def get_action(self, obs, info):
        occ = obs["map_tensor"][0]
        pos = np.array([info["pos_x"], info["pos_y"]])
        cur_alt = info["alt_idx"]
        cur_cov = info.get(f"cov_alt{cur_alt}", 0)

        # Detect if stuck (coverage not increasing)
        if abs(cur_cov - self.prev_cov) < 0.001:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        self.prev_cov = cur_cov

        # Find frontiers at current altitude
        fronts = []
        for r in range(1, occ.shape[0] - 1, 2):
            for c in range(1, occ.shape[1] - 1, 2):
                if occ[r, c] < 0.25:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if 0.4 < occ[r + dr, c + dc] < 0.6:
                            fronts.append(np.array([r, c], np.float32))
                            break

        # Switch altitude if: no frontiers OR stuck for 15+ steps
        if len(fronts) == 0 or self.stuck_steps > 15:
            # Find least explored altitude
            coverages = [info.get(f"cov_alt{i}", 0) for i in range(N_ALTITUDES)]
            least = int(np.argmin(coverages))
            if least != cur_alt:
                self.stuck_steps = 0
                dz = 0.8 if least > cur_alt else -0.8
                return np.array([np.random.uniform(-0.3, 0.3),
                               np.random.uniform(-0.3, 0.3), dz], np.float32)

        if len(fronts) == 0:
            return np.random.uniform(-1, 1, 3).astype(np.float32)

        dists = [np.linalg.norm(f - pos) for f in fronts]
        nearest = fronts[np.argmin(dists)]
        d = nearest - pos
        n = np.linalg.norm(d)
        if n > 0.01: d = d / n
        noise = np.random.normal(0, 0.1, 2)
        dx, dy = np.clip(d + noise, -1, 1)
        return np.array([dx, dy, 0.0], np.float32)


class Spiral3D:
    """Lawnmower sweep at each altitude sequentially."""
    name = "Spiral"
    def __init__(self, grid_size=48, n_alt=4):
        self.grid_size = grid_size
        self.n_alt = n_alt
        self.target_alt = 0
        self.wps = self._gen_wps()
        self.wp_idx = 0
        self.stuck = 0

    def _gen_wps(self):
        wps = []
        sp = 8
        gs = self.grid_size
        for col in range(5, gs - 5, sp):
            if (col // sp) % 2 == 0:
                for row in range(5, gs - 5, 4):
                    wps.append(np.array([row, col], np.float32))
            else:
                for row in range(gs - 6, 4, -4):
                    wps.append(np.array([row, col], np.float32))
        return wps

    def reset(self):
        self.target_alt = 0
        self.wp_idx = 0
        self.stuck = 0

    def get_action(self, obs, info):
        pos = np.array([info["pos_x"], info["pos_y"]])
        cur_alt = info["alt_idx"]

        if cur_alt != self.target_alt:
            dz = 0.8 if self.target_alt > cur_alt else -0.8
            return np.array([0.0, 0.0, dz], np.float32)

        if self.wp_idx >= len(self.wps) or self.stuck > 20:
            self.target_alt = min(self.target_alt + 1, self.n_alt - 1)
            self.wp_idx = 0
            self.stuck = 0
            if self.target_alt >= self.n_alt:
                return np.random.uniform(-1, 1, 3).astype(np.float32)
            return np.array([0.0, 0.0, 0.8], np.float32)

        target = self.wps[self.wp_idx]
        d = target - pos
        dist = np.linalg.norm(d)
        if dist < 3.0:
            self.wp_idx += 1
            self.stuck = 0
            if self.wp_idx >= len(self.wps):
                return np.array([0.0, 0.0, 0.0], np.float32)
            d = self.wps[self.wp_idx] - pos
            dist = np.linalg.norm(d)
        if dist > 0.01:
            d = d / dist
        else:
            self.stuck += 1
        if info.get("collision", False):
            self.stuck += 3
            self.wp_idx += 1
        return np.array([d[0], d[1], 0.0], np.float32)


class PotentialField3D:
    """Info-theoretic potential field with altitude awareness."""
    name = "Potential Field"
    def __init__(self, n_headings=16):
        self.n_headings = n_headings
    def reset(self): pass

    def get_action(self, obs, info):
        occ = obs["map_tensor"][0]
        vis = obs["map_tensor"][1]
        scalars = obs["scalars"]
        pos = np.array([info["pos_x"], info["pos_y"]])
        cov = scalars[0] * 4.0
        gs = occ.shape[0]

        best_score, best_act = -float("inf"), np.zeros(3, np.float32)

        for i in range(self.n_headings):
            ang = 2 * math.pi * i / self.n_headings
            d = np.array([math.cos(ang), math.sin(ang)])
            ig, od, rp = 0.0, float("inf"), 0.0

            for dist in range(1, 20):
                r, c = int(pos[0] + d[0] * dist), int(pos[1] + d[1] * dist)
                if 0 <= r < gs and 0 <= c < gs:
                    if occ[r, c] > 0.8:
                        od = dist; break
                    elif 0.4 < occ[r, c] < 0.6:
                        ig += 1.0
                    if vis[r, c] > 0.5:
                        rp += 0.2
                else:
                    break

            score = ig - rp
            if od < 3:
                score -= 10.0
            if cov > 2.0:
                score -= ig * 0.5
            if score > best_score:
                best_score = score
                best_act[:2] = d.astype(np.float32)

        # Altitude: switch if current alt well-explored but others aren't
        cur_cov = info.get(f"cov_alt{info['alt_idx']}", 0)
        coverages = [info.get(f"cov_alt{i}", 0) for i in range(N_ALTITUDES)]
        least = int(np.argmin(coverages))
        if cur_cov > 0.3 and coverages[least] < cur_cov * 0.5 and least != info["alt_idx"]:
            best_act[2] = 0.8 if least > info["alt_idx"] else -0.8

        noise = np.random.normal(0, 0.05, 3).astype(np.float32)
        return np.clip(best_act + noise, -1, 1)


class GreedyInfoGain3D:
    """Always pick the action that reveals maximum new cells. No uncertainty awareness.
    This is the key comparison: greedy coverage vs uncertainty-aware RL."""
    name = "Greedy Info Gain"
    def __init__(self, n_samples=12):
        self.n_samples = n_samples
        self.stuck = 0
        self.prev_cov = 0.0
    def reset(self):
        self.stuck = 0
        self.prev_cov = 0.0

    def get_action(self, obs, info):
        occ = obs["map_tensor"][0]
        pos = np.array([info["pos_x"], info["pos_y"]])
        gs = occ.shape[0]

        # Detect stagnation for altitude switching
        cur_cov = info.get(f"cov_alt{info['alt_idx']}", 0)
        if abs(cur_cov - self.prev_cov) < 0.001:
            self.stuck += 1
        else:
            self.stuck = 0
        self.prev_cov = cur_cov

        # Switch altitude if stuck
        if self.stuck > 20:
            coverages = [info.get(f"cov_alt{i}", 0) for i in range(N_ALTITUDES)]
            least = int(np.argmin(coverages))
            if least != info["alt_idx"]:
                self.stuck = 0
                dz = 0.8 if least > info["alt_idx"] else -0.8
                return np.array([np.random.uniform(-0.3, 0.3),
                               np.random.uniform(-0.3, 0.3), dz], np.float32)

        # Sample candidate actions, pick the one pointing toward most unknown cells
        best_score = -1
        best_act = np.random.uniform(-1, 1, 3).astype(np.float32)

        for i in range(self.n_samples):
            angle = 2 * math.pi * i / self.n_samples
            d = np.array([math.cos(angle), math.sin(angle)])
            unknown_count = 0

            for dist in range(1, 20):
                r = int(pos[0] + d[0] * dist)
                c = int(pos[1] + d[1] * dist)
                if 0 <= r < gs and 0 <= c < gs:
                    if occ[r, c] > 0.8:  # obstacle
                        break
                    elif 0.4 < occ[r, c] < 0.6:  # unknown
                        unknown_count += 1
                else:
                    break

            if unknown_count > best_score:
                best_score = unknown_count
                best_act = np.array([d[0], d[1], 0.0], np.float32)

        noise = np.random.normal(0, 0.08, 3).astype(np.float32)
        return np.clip(best_act + noise, -1, 1)


class RRTExplorer3D:
    """RRT-based exploration: sample random reachable points, navigate to the one
    with highest exploration value (most unknown cells nearby).
    Based on Umari & Mukhopadhyay (2017) adapted for altitude-aware exploration."""
    name = "RRT Explorer"
    def __init__(self, n_samples=50, grid_size=48):
        self.n_samples = n_samples
        self.gs = grid_size
        self.target = None
        self.stuck = 0
        self.prev_cov = 0.0

    def reset(self):
        self.target = None
        self.stuck = 0
        self.prev_cov = 0.0

    def get_action(self, obs, info):
        occ = obs["map_tensor"][0]
        pos = np.array([info["pos_x"], info["pos_y"]])

        # Detect stagnation for altitude switching
        cur_cov = info.get(f"cov_alt{info['alt_idx']}", 0)
        if abs(cur_cov - self.prev_cov) < 0.001:
            self.stuck += 1
        else:
            self.stuck = 0
        self.prev_cov = cur_cov

        if self.stuck > 25:
            coverages = [info.get(f"cov_alt{i}", 0) for i in range(N_ALTITUDES)]
            least = int(np.argmin(coverages))
            if least != info["alt_idx"]:
                self.stuck = 0
                self.target = None
                dz = 0.8 if least > info["alt_idx"] else -0.8
                return np.array([0.0, 0.0, dz], np.float32)

        # Check if we reached current target or need a new one
        if self.target is not None:
            dist_to_target = np.linalg.norm(self.target - pos)
            if dist_to_target < 3.0:
                self.target = None  # reached, pick new target

        # Sample new target using RRT-style random sampling
        if self.target is None:
            best_score = -1
            best_point = None

            for _ in range(self.n_samples):
                # Random point in the grid
                r = np.random.randint(2, self.gs - 2)
                c = np.random.randint(2, self.gs - 2)

                # Must be in free space (known and not occupied)
                if occ[r, c] > 0.25:  # not free
                    continue

                # Score: count unknown cells in neighborhood
                rlo = max(0, r - 5)
                rhi = min(self.gs, r + 6)
                clo = max(0, c - 5)
                chi = min(self.gs, c + 6)
                patch = occ[rlo:rhi, clo:chi]
                unknown_nearby = np.sum((patch > 0.4) & (patch < 0.6))

                # Check reachability: rough line-of-sight check
                d = np.array([r, c], np.float32) - pos
                n = np.linalg.norm(d)
                blocked_steps = 0
                checked_steps = 0
                if n > 1:
                    for step in range(1, int(n) + 1):
                        p = pos + (step / n) * d
                        pr, pc = int(p[0]), int(p[1])
                        if 0 <= pr < self.gs and 0 <= pc < self.gs:
                            checked_steps += 1
                            if occ[pr, pc] > 0.8:
                                blocked_steps += 1

                blocked = checked_steps > 0 and blocked_steps > (checked_steps // 2)

                if not blocked and unknown_nearby > best_score:
                    best_score = unknown_nearby
                    best_point = np.array([r, c], np.float32)

            if best_point is not None:
                self.target = best_point
            else:
                # Fallback: random direction
                return np.random.uniform(-1, 1, 3).astype(np.float32)

        # Navigate toward target
        d = self.target - pos
        n = np.linalg.norm(d)
        if n > 0.01:
            d = d / n
        noise = np.random.normal(0, 0.05, 2)
        dx, dy = np.clip(d + noise, -1, 1)
        return np.array([dx, dy, 0.0], np.float32)


def get_all_baselines():
    return [RandomWalk3D(), NearestFrontier3D(), Spiral3D(),
            PotentialField3D(), GreedyInfoGain3D(), RRTExplorer3D()]
