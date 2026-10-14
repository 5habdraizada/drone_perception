"""Search environment for learning (or benchmarking) where to fly to uncover hidden things.

Gymnasium-style API without the dependency:
    env = DroneSearchEnv(SearchConfig(), seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Each episode is a random scene: roofs, tree canopy, and objects hidden on the ground under trees.
Per step the drone moves step_m in one of 8 directions at one of two altitudes; oracle perception
feeds the map and the BeliefMap.

Reward (bits, see perception/belief.py):
    entropy removed this step (hidden objects)
  + discovery_bonus per newly confirmed detection that matches a real hidden object
  - false_alarm_penalty per confirmed detection that matches nothing
  - energy_cost per metre flown
  - crash_penalty (and the episode ends) if the drone flies into something taller than its altitude
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from perception.belief import BeliefConfig, BeliefMap
from perception.mapping import GridMap, MapConfig
from perception.terrain import StereoRig
from sim.oracle_perception import OraclePerception
from sim.synthetic_stereo import Block, Scene

RIG = StereoRig(fx=577.0, fy=577.0, cx=320.0, cy=176.0, baseline=0.10)
SIZE = (640, 352)
DIRECTIONS = [(math.cos(a), math.sin(a)) for a in np.radians(np.arange(0, 360, 45))]


@dataclass
class SearchConfig:
    area: tuple[float, float, float, float] = (-30.0, 30.0, -20.0, 20.0)
    altitudes: tuple[float, float] = (20.0, 8.0)   # cruise, low (m above ground)
    step_m: float = 5.0
    max_steps: int = 150
    map_resolution: float = 1.0
    discovery_bonus: float = 20.0
    false_alarm_penalty: float = 10.0
    energy_cost: float = 0.05                       # bits per metre flown
    crash_penalty: float = 100.0
    match_radius: float = 3.0                       # m: a detection this close to a target found it
    window: int = 32                                # observation: cells per side, egocentric
    n_roofs: tuple[int, int] = (1, 3)
    n_tree_patches: tuple[int, int] = (2, 4)
    n_hidden: tuple[int, int] = (1, 2)


@dataclass
class Episode:
    scene: Scene
    hidden: list[Block]
    reported: list = field(default_factory=list)   # detections already scored
    found: set = field(default_factory=set)        # indices of hidden objects found
    false_alarms: int = 0


class DroneSearchEnv:
    n_actions = 16  # 8 directions x 2 altitudes

    def __init__(self, config: SearchConfig | None = None, seed: int = 0):
        self.cfg = config or SearchConfig()
        self.rng = np.random.default_rng(seed)

    # -- API --------------------------------------------------------------------

    def reset(self, scene: Episode | None = None):
        cfg = self.cfg
        self.ep = scene or random_episode(cfg, self.rng)
        self.oracle = OraclePerception(self.ep.scene, RIG, SIZE, seed=int(self.rng.integers(1 << 31)))
        self.grid = GridMap(RIG, SIZE, MapConfig(resolution=cfg.map_resolution))
        self.grid.ensure_area(*cfg.area)
        self.belief = BeliefMap(self.grid, RIG, SIZE, BeliefConfig())
        x0, _, y0, _ = cfg.area
        self.x, self.y, self.alt_i = x0 + 2.0, y0 + 2.0, 0
        self.t, self.flown, self.crashed = 0, 0.0, False
        self._observe(self.x, self.y)
        self._entropy = self.belief.entropy(cfg.area)["object"]
        self.initial_entropy = self._entropy
        return self.observation(), self.info()

    def step(self, action: int):
        cfg = self.cfg
        d, alt_i = DIRECTIONS[action % 8], action // 8
        nx = float(np.clip(self.x + d[0] * cfg.step_m, cfg.area[0], cfg.area[1]))
        ny = float(np.clip(self.y + d[1] * cfg.step_m, cfg.area[2], cfg.area[3]))
        dist = math.hypot(nx - self.x, ny - self.y) + abs(cfg.altitudes[alt_i] - self.agl)
        reward = -cfg.energy_cost * dist
        if self.obstacle_height(nx, ny, self.x, self.y) + 1.0 > cfg.altitudes[alt_i]:
            self.crashed = True
            return self.observation(), reward - cfg.crash_penalty, True, False, self.info()

        self.x, self.y, self.alt_i = nx, ny, alt_i
        self.flown += dist
        self.t += 1
        self._observe(nx, ny)

        h = self.belief.entropy(cfg.area)["object"]
        reward += self._entropy - h
        self._entropy = h
        reward += self._score_detections()
        return self.observation(), reward, False, self.t >= cfg.max_steps, self.info()

    # -- helpers -------------------------------------------------------------------

    @property
    def agl(self) -> float:
        return self.cfg.altitudes[self.alt_i]

    def obstacle_height(self, x1, y1, x0=None, y0=None) -> float:
        """Tallest true surface along the move (the simulator knows the scene)."""
        x0 = x1 if x0 is None else x0
        y0 = y1 if y0 is None else y0
        tallest = 0.0
        for s in np.linspace(0.0, 1.0, 6):
            px, py = x0 + (x1 - x0) * s, y0 + (y1 - y0) * s
            for b in self.ep.scene.blocks:
                if b.h > 0 and abs(px - b.x) < b.w / 2 + 1.0 and abs(py - b.y) < b.l / 2 + 1.0:
                    tallest = max(tallest, b.h)
        return tallest

    def observation(self) -> dict:
        """Egocentric window of belief layers around the drone (window x window map cells),
        plus scalars. Outside the map is filled with neutral values."""
        g, w = self.grid, self.cfg.window
        r0, c0 = g.world_to_cell(self.x, self.y)
        rr, cc = np.mgrid[r0 - w // 2: r0 + w // 2, c0 - w // 2: c0 + w // 2]
        ok = (rr >= 0) & (rr < g.shape[0]) & (cc >= 0) & (cc < g.shape[1])

        def win(layer, fill):
            out = np.full((w, w), fill, np.float32)
            out[ok] = layer[rr[ok], cc[ok]]
            return out

        b = self.belief
        return {"p_object": win(b.p_object, 0.0), "view_quality": win(b.view_quality, 1.0),
                "height": win(np.nan_to_num(g.height, nan=0.0), 0.0),
                "searchable": win(b.searchable.astype(np.float32), 0.0),
                "state": np.array([self.alt_i, self.t / self.cfg.max_steps,
                                   (self.x - self.cfg.area[0]) / (self.cfg.area[1] - self.cfg.area[0]),
                                   (self.y - self.cfg.area[2]) / (self.cfg.area[3] - self.cfg.area[2])], np.float32)}

    def info(self) -> dict:
        return {"entropy": self._entropy if hasattr(self, "_entropy") else None, "flown_m": self.flown,
                "found": len(self.ep.found), "targets": len(self.ep.hidden),
                "false_alarms": self.ep.false_alarms, "crashed": self.crashed, "step": self.t}

    def _observe(self, x, y) -> None:
        f, pose = self.oracle.frame(x, y, self.agl, float(self.t)), self.oracle.pose(x, y, self.agl)
        self.grid.update(f, pose)
        self.belief.observe(f, pose)

    def _score_detections(self) -> float:
        """Score detections not scored before: the first one on each hidden object earns the bonus,
        later ones on a found object are ignored, ones on nothing are false alarms."""
        cfg, bonus = self.cfg, 0.0
        for det in self.belief.detections("object"):
            if any(math.hypot(det["x"] - fx, det["y"] - fy) < cfg.match_radius for fx, fy in self.ep.reported):
                continue
            self.ep.reported.append((det["x"], det["y"]))
            hits = {i for i, h in enumerate(self.ep.hidden) if math.hypot(det["x"] - h.x, det["y"] - h.y) < cfg.match_radius}
            if not hits:
                self.ep.false_alarms += 1
                bonus -= cfg.false_alarm_penalty
            elif not hits <= self.ep.found:
                self.ep.found |= hits
                bonus += cfg.discovery_bonus
        return bonus


def random_episode(cfg: SearchConfig, rng: np.random.Generator) -> Episode:
    x0, x1, y0, y1 = cfg.area
    blocks, hidden = [], []
    rand = lambda lo, hi: float(rng.uniform(lo, hi))
    count = lambda rng_pair: int(rng.integers(rng_pair[0], rng_pair[1] + 1))
    for _ in range(count(cfg.n_roofs)):
        blocks.append(Block(rand(x0 + 8, x1 - 8), rand(y0 + 8, y1 - 8), rand(6, 12), rand(6, 12), rand(5, 12)))
    patches = []
    for _ in range(count(cfg.n_tree_patches)):
        p = Block(rand(x0 + 6, x1 - 6), rand(y0 + 6, y1 - 6), rand(6, 12), rand(6, 12), rand(4, 6), pattern="canopy")
        blocks.append(p)
        patches.append(p)
    for _ in range(count(cfg.n_hidden)):
        p = patches[int(rng.integers(len(patches)))]
        h = Block(p.x + rand(-p.w / 4, p.w / 4), p.y + rand(-p.l / 4, p.l / 4), 1.2, 1.2, 0.0, pattern="hidden")
        blocks.append(h)
        hidden.append(h)
    extent = 2 * max(abs(x0), abs(x1), abs(y0), abs(y1)) + 20
    return Episode(Scene(extent=extent, res=0.1, blocks=blocks), hidden)
