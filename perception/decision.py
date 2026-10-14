"""Mission planner: map + drone state -> one waypoint command per cycle.

Priorities: return/land > investigate > verify > explore. Every altitude clears the tallest
thing along the path; unmapped cells are assumed unknown_height tall. An optional advisor
(perception/advisor.py) re-ranks and dismisses anomalies against a plain-language brief.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from perception.mapping import GridMap


@dataclass
class DecisionConfig:
    cruise_agl: float = 8.0          # m above ground while exploring / transiting
    investigate_agl: float = 4.0     # m above ground for a closer look
    min_clearance: float = 2.0       # m kept above the tallest thing along the path
    unknown_height: float = 3.0      # m assumed for unmapped cells (downward cameras can't see ahead)
    clearance_radius: float = 1.5    # m around the path checked for obstacles
    arrive_radius: float = 1.0       # m: close enough to a target
    investigate_hold: float = 5.0    # s hovering over an anomaly
    match_radius: float = 1.5        # m: anomalies this close are the same one

    area: tuple[float, float, float, float] | None = None  # (xmin, xmax, ymin, ymax) to search
    explore_distance_ref: float = 6.0   # m; frontier utility halves at this distance
    min_frontier_cells: int = 4      # ignore frontier scraps smaller than this
    anomaly_distance_ref: float = 15.0  # m; anomaly utility halves at this distance
    verify_candidates: bool = True
    min_relevance: float = 0.3       # advisor: anomalies judged less relevant to the brief are dismissed
    unassessed_relevance: float = 1.0  # advisor: priority weight until its answer arrives (don't wait)

    battery_per_m: float = 0.001     # battery fraction used per metre flown
    battery_reserve: float = 0.15    # fraction kept in hand on arrival home
    max_pose_sigma: float = 3.0      # m; worse than this and the map can't be trusted -> go home


@dataclass
class Command:
    mode: str                        # "explore", "investigate", "verify", "return", "land"
    x: float
    y: float
    agl: float                       # target altitude above ground, m
    hold: float = 0.0                # s to hover once there
    reason: str = ""
    target: dict | None = None       # the anomaly being investigated / verified


@dataclass
class _Goal:
    mode: str
    x: float
    y: float
    target: dict | None = None
    arrived_at: float | None = None


@dataclass
class MissionPlanner:
    home: tuple[float, float]
    cfg: DecisionConfig = field(default_factory=DecisionConfig)
    report: list[dict] = field(default_factory=list)   # anomalies investigated, with when and what was seen
    # Optional mission steering (perception/advisor.py): an AsyncAdvisor judging anomaly crops
    # against a brief, and crops(anomaly) -> image | None. It only re-ranks and dismisses
    # anomalies; safety (return, landing, clearance) never consults it.
    advisor: object | None = None
    crops: Callable[[dict], np.ndarray | None] | None = None
    dismissed: list[dict] = field(default_factory=list)  # anomalies the advisor ruled out, with its reason

    def __post_init__(self):
        self._goal: _Goal | None = None
        self._verified: list[tuple[float, float]] = []
        self._landing = False
        self._keys: list[tuple[float, float, str]] = []

    def decide(self, grid: GridMap, x: float, y: float, agl: float, battery: float,
               pose_sigma: float, t: float) -> Command:
        """battery: remaining fraction 0..1; pose_sigma: horizontal position uncertainty (m)."""
        cfg = self.cfg
        if cfg.area is not None:
            grid.ensure_area(*cfg.area)

        # 1. Safety first.
        to_home = math.hypot(x - self.home[0], y - self.home[1])
        if self._landing or (self._goal and self._goal.mode == "return" and to_home <= cfg.arrive_radius):
            self._landing = True
            return Command("land", *self.home, 0.0, reason="home")
        if pose_sigma > cfg.max_pose_sigma:
            return self._return(grid, x, y, f"position uncertainty {pose_sigma:.1f} m > {cfg.max_pose_sigma} m")
        needed = to_home * cfg.battery_per_m + cfg.battery_reserve
        if battery <= needed:
            return self._return(grid, x, y, f"battery {battery:.0%}, {needed:.0%} needed to get home")
        if self._goal and self._goal.mode == "return":
            return self._return(grid, x, y, "returning")

        # Finish the current goal unless a newly confirmed anomaly should interrupt exploring.
        pending = self._pending_anomalies(grid)
        if self._goal and not (self._goal.mode in ("explore", "verify") and pending):
            cmd = self._continue(grid, x, y, t)
            if cmd is not None:
                return cmd

        # 2. Investigate.
        if pending:
            best = max(pending, key=lambda a: self._anomaly_utility(a, x, y))
            self._goal = _Goal("investigate", best["x"], best["y"], best)
            return self._go(grid, x, y, "investigate", best["x"], best["y"], cfg.investigate_agl,
                            f"anomaly at ({best['x']:.1f}, {best['y']:.1f}), peak z {best['peak_novelty']:.1f}",
                            best)

        # 3. Verify single-frame candidates.
        if cfg.verify_candidates:
            cands = self._candidates(grid)
            if cands:
                best = max(cands, key=lambda a: self._anomaly_utility(a, x, y))
                self._goal = _Goal("verify", best["x"], best["y"], best)
                return self._go(grid, x, y, "verify", best["x"], best["y"], cfg.cruise_agl,
                                "seen as unusual once: look again", best)

        # 4. Explore.
        frontier = self._frontier_target(grid, x, y)
        if frontier is not None:
            fx, fy, size = frontier
            self._goal = _Goal("explore", fx, fy)
            return self._go(grid, x, y, "explore", fx, fy, cfg.cruise_agl, f"frontier of {size} unmapped cells")

        return self._return(grid, x, y, "search area covered")

    def _continue(self, grid, x, y, t) -> Command | None:
        g, cfg = self._goal, self.cfg
        reached = math.hypot(x - g.x, y - g.y) <= cfg.arrive_radius
        if g.mode == "investigate":
            if g.target is not None and self._dismiss_if_irrelevant(g.target, t):
                self._goal = None  # the advisor's answer arrived on the way: not worth it
                return None
            if reached and g.arrived_at is None:
                g.arrived_at = t
            if g.arrived_at is not None and t - g.arrived_at >= cfg.investigate_hold:
                self._finish_investigation(grid, g, t)
                return None
            hold = cfg.investigate_hold if reached else 0.0
            return self._go(grid, x, y, "investigate", g.x, g.y, cfg.investigate_agl,
                            "holding over anomaly" if reached else "flying to anomaly", g.target, hold)
        if g.mode == "verify":
            if reached:
                self._verified.append((g.x, g.y))
                self._goal = None
                return None
            return self._go(grid, x, y, "verify", g.x, g.y, cfg.cruise_agl, "looking again", g.target)
        if g.mode == "explore":
            seen_now = self._seen_at(grid, g.x, g.y)
            if reached or seen_now:  # the target got mapped on the way: pick a new one
                self._goal = None
                return None
            return self._go(grid, x, y, "explore", g.x, g.y, cfg.cruise_agl, "exploring")
        return None

    def _finish_investigation(self, grid: GridMap, g: _Goal, t: float) -> None:
        # Re-read the anomaly from the map: the close look has refined it.
        latest = min(grid.anomalies(), key=lambda a: math.hypot(a["x"] - g.x, a["y"] - g.y), default=g.target)
        if math.hypot(latest["x"] - g.x, latest["y"] - g.y) > self.cfg.match_radius:
            latest = g.target
        entry = {**latest, "investigated_at": t}
        advice = self._advice(g.target)
        if advice is not None:
            entry["advice"] = advice.__dict__
        self.report.append(entry)
        self._goal = None

    def _return(self, grid, x, y, reason: str) -> Command:
        self._goal = _Goal("return", *self.home)
        return self._go(grid, x, y, "return", *self.home, self.cfg.cruise_agl, reason)

    def _go(self, grid, x, y, mode, tx, ty, agl, reason, target=None, hold=0.0) -> Command:
        safe = self.path_clearance(grid, x, y, tx, ty) + self.cfg.min_clearance
        return Command(mode, float(tx), float(ty), float(max(agl, safe)), hold, reason, target)

    def path_clearance(self, grid: GridMap, x0: float, y0: float, x1: float, y1: float) -> float:
        """Tallest mapped surface near the straight path (unknown_height where nothing is mapped)."""
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / grid.cfg.resolution) + 1)
        tallest = 0.0
        for s in np.linspace(0.0, 1.0, n):
            h = grid.clearance(x0 + s * (x1 - x0), y0 + s * (y1 - y0), self.cfg.clearance_radius)
            tallest = max(tallest, self.cfg.unknown_height if math.isnan(h) else h)
        return tallest

    def _pending_anomalies(self, grid: GridMap) -> list[dict]:
        done = [(r["x"], r["y"]) for r in self.report] + [(d["x"], d["y"]) for d in self.dismissed]
        out = []
        for a in grid.anomalies():
            if self._goal and self._goal.mode == "investigate" and self._near(a, [(self._goal.x, self._goal.y)]):
                continue
            if self._near(a, done) or not self._in_area(a["x"], a["y"]):
                continue
            if self._dismiss_if_irrelevant(a, None):
                continue
            out.append(a)
        return out

    def _advice(self, a: dict):
        """The advisor's assessment of this anomaly, requesting one if needed; None until it arrives."""
        if self.advisor is None:
            return None
        key = self._key(a)
        res = self.advisor.result(key)
        if res is None and self.crops is not None:
            crop = self.crops(a)
            if crop is not None:
                self.advisor.request(key, crop)
        return res

    def _dismiss_if_irrelevant(self, a: dict, t: float | None) -> bool:
        res = self._advice(a)
        if res is None or (res.investigate and res.relevance >= self.cfg.min_relevance):
            return False
        if not self._near(a, [(d["x"], d["y"]) for d in self.dismissed]):
            self.dismissed.append({**a, "advice": res.__dict__, "dismissed_at": t})
        return True

    def _key(self, a: dict) -> str:
        """A stable id per anomaly: the map refines positions slightly as it learns."""
        for kx, ky, key in self._keys:
            if math.hypot(a["x"] - kx, a["y"] - ky) <= self.cfg.match_radius:
                return key
        key = f"anomaly-{len(self._keys)}"
        self._keys.append((a["x"], a["y"], key))
        return key

    def _candidates(self, grid: GridMap) -> list[dict]:
        confirmed = [(a["x"], a["y"]) for a in grid.anomalies()]
        return [a for a in grid.anomalies(min_hits=1, min_cells=1)
                if a["hits"] < grid.cfg.min_hits and not self._near(a, confirmed)
                and not self._near(a, self._verified) and self._in_area(a["x"], a["y"])]

    def _frontier_target(self, grid: GridMap, x: float, y: float):
        """Best (x, y, size) to explore: the edge of the mapped area, or, when nothing borders
        the unmapped part of the search area (e.g. before take-off), that part itself."""
        if grid.origin is None:
            return (x, y, 0) if self.cfg.area is not None else None  # nothing mapped yet: take off here
        area = grid.area_mask(*self.cfg.area) if self.cfg.area is not None else None
        mask = grid.frontier() if area is None else grid.frontier() & area
        if mask.sum() < self.cfg.min_frontier_cells and area is not None:
            mask = area & (grid.seen == 0)
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        best, best_u = None, -1.0
        for k in range(1, n):
            rr, cc = np.nonzero(labels == k)
            if len(rr) < self.cfg.min_frontier_cells:
                continue
            wx, wy = grid.cell_to_world(rr, cc)
            d = np.hypot(wx - x, wy - y)
            i = int(np.argmin(d))
            u = math.sqrt(len(rr)) / (1.0 + d[i] / self.cfg.explore_distance_ref)
            if u > best_u:
                best, best_u = (float(wx[i]), float(wy[i]), len(rr)), u
        return best

    def _anomaly_utility(self, a: dict, x: float, y: float) -> float:
        d = math.hypot(a["x"] - x, a["y"] - y)
        res = self._advice(a) if self.advisor is not None else None
        relevance = res.relevance if res is not None else self.cfg.unassessed_relevance
        return relevance * a["peak_novelty"] * a["hits"] / (1.0 + d / self.cfg.anomaly_distance_ref)

    def _seen_at(self, grid: GridMap, x: float, y: float) -> bool:
        r, c = grid.world_to_cell(x, y)
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid.seen[r, c] > 0

    def _near(self, a: dict, points) -> bool:
        return any(math.hypot(a["x"] - px, a["y"] - py) <= self.cfg.match_radius for px, py in points)

    def _in_area(self, x: float, y: float) -> bool:
        if self.cfg.area is None:
            return True
        x0, x1, y0, y1 = self.cfg.area
        return x0 <= x <= x1 and y0 <= y <= y1
