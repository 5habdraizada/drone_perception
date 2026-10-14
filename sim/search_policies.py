"""Policies for DroneSearchEnv: baselines and the information-greedy planner the reward is meant to drive."""
from __future__ import annotations

import math

import numpy as np

from perception.belief import _entropy
from sim.search_env import DIRECTIONS, DroneSearchEnv


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: DroneSearchEnv) -> int:
        safe = [a for a in range(env.n_actions) if _safe(env, a)]
        return int(self.rng.choice(safe or [a for a in range(8)]))


class LawnmowerPolicy:
    """Back-and-forth sweep at cruise altitude, lanes spaced by the camera footprint."""

    def __init__(self):
        self.waypoints, self.i = None, 0

    def __call__(self, env: DroneSearchEnv) -> int:
        if self.waypoints is None:
            x0, x1, y0, y1 = env.cfg.area
            lane = 2 * env.cfg.altitudes[0] * 176 / 577 * 0.9
            ys = np.arange(y0 + lane / 2, y1, lane)
            self.waypoints = [(x, y) for k, y in enumerate(ys) for x in ((x0 + 2, x1 - 2) if k % 2 == 0 else (x1 - 2, x0 + 2))]
        while self.i < len(self.waypoints) and math.hypot(self.waypoints[self.i][0] - env.x,
                                                          self.waypoints[self.i][1] - env.y) < env.cfg.step_m:
            self.i += 1
        tx, ty = self.waypoints[min(self.i, len(self.waypoints) - 1)]
        return _towards(env, tx, ty, alt_i=0)


class GreedyInfoPolicy:
    """Pick the move whose expected information gain minus energy is highest.
    Safety uses only what the drone knows: mapped heights, unmapped cells assumed unknown_height tall."""

    def __init__(self, unknown_height: float = 12.0, min_gain: float = 0.5, margin: float = 2.5):
        self.unknown_height, self.min_gain, self.margin = unknown_height, min_gain, margin

    def __call__(self, env: DroneSearchEnv) -> int:
        cfg, b = env.cfg, env.belief
        best, best_score = None, -math.inf
        for a in range(env.n_actions):
            nx, ny, alt = _target(env, a)
            if not self._known_safe(env, nx, ny, alt):
                continue
            dist = math.hypot(nx - env.x, ny - env.y) + abs(alt - env.agl)
            score = b.expected_gain(nx, ny, alt) -cfg.energy_cost * dist
            if score > best_score:
                best, best_score = a, score
        if best is not None and best_score > self.min_gain:
            return best
        # Nothing worth much nearby: head for the most uncertain part of the area.
        tx, ty = _most_uncertain(env)
        return _towards(env, tx, ty, alt_i=0)

    def _known_safe(self, env, nx, ny, alt) -> bool:
        """Every map cell within margin of the path is mapped and at least margin below alt
        (an unmapped cell could be anything up to unknown_height)."""
        g = env.grid
        k = int(math.ceil(self.margin / g.cfg.resolution))
        for s in np.linspace(0, 1, 6):
            r, c = g.world_to_cell(env.x + (nx - env.x) * s, env.y + (ny - env.y) * s)
            patch = g.height[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
            h = np.where(np.isnan(patch), self.unknown_height, patch)
            if h.size == 0 or h.max() + self.margin > alt:
                return False
        return True


def _target(env, a):
    d, alt_i = DIRECTIONS[a % 8], a // 8
    c = env.cfg
    return (float(np.clip(env.x + d[0] * c.step_m, c.area[0], c.area[1])),
            float(np.clip(env.y + d[1] * c.step_m, c.area[2], c.area[3])), c.altitudes[alt_i])


def _safe(env, a) -> bool:
    nx, ny, alt = _target(env, a)
    return env.obstacle_height(nx, ny, env.x, env.y) + 1.0 <= alt


def _towards(env, tx, ty, alt_i) -> int:
    ang = math.atan2(ty - env.y, tx - env.x)
    k = int(round(ang / (math.pi / 4))) % 8
    return k + 8 * alt_i


def _most_uncertain(env, block_m: float = 10.0):
    """Centre of the block of the search area with the most remaining entropy per metre of travel."""
    g, b, c = env.grid, env.belief, env.cfg
    h = _entropy(b.p_object) * b.searchable
    mask = g.area_mask(*c.area)
    best, best_v = (env.x, env.y), -1.0
    for bx in np.arange(c.area[0] + block_m / 2, c.area[1], block_m):
        for by in np.arange(c.area[2] + block_m / 2, c.area[3], block_m):
            r0, c0 = g.world_to_cell(bx - block_m / 2, by - block_m / 2)
            r1, c1 = g.world_to_cell(bx + block_m / 2, by + block_m / 2)
            v = float((h * mask)[max(r0, 0):r1, max(c0, 0):c1].sum()) / (1 + math.hypot(bx - env.x, by - env.y) / 20)
            if v > best_v:
                best, best_v = (bx, by), v
    return best


def run_episode(env: DroneSearchEnv, policy, episode=None) -> dict:
    """Run one episode; returns env.info() plus total reward, share of entropy removed and the flown path."""
    env.reset(episode)
    total, done, trunc = 0.0, False, False
    path = [(env.x, env.y, env.agl)]
    while not (done or trunc):
        _, r, done, trunc, _ = env.step(policy(env))
        total += r
        path.append((env.x, env.y, env.agl))
    info = env.info()
    info["reward"] = total
    info["entropy_removed"] = 1 - info["entropy"] / env.initial_entropy
    info["path"] = path
    return info
