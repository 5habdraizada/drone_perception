"""Belief about what can't be seen directly, and the reward for reducing uncertainty about it.

Two hidden-target hypotheses per map cell, as log-odds on the GridMap's grid:
  object      something of interest at ground level, possibly hidden under tree canopy.
              Clear views of the ground without a detection lower it; novelty hits raise it.
              Canopy lets only part of a view through (gaps), so low passes over trees count more.
              Roofs (smooth raised surfaces) are not searchable: nothing at ground level hides there.
  subsurface  a buried void/structure, from indirect visual cues: ground depressions raise it.

Uncertainty = binary entropy of each cell's probability, summed over the area (bits).
  planner  expected_gain(x, y, agl): approximate entropy a view from there would remove
  RL       step reward = entropy removed + discovery bonus - costs (see sim/search_env.py)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from perception.mapping import GridMap
from perception.terrain import StereoRig


@dataclass
class BeliefConfig:
    object_prior: float = 0.02       # chance a searchable cell hides an object of interest
    clear_view: float = -4.0         # object log-odds per unit *improvement* in the best ground view
    hit: float = 2.5                 # object log-odds for a strong novelty hit in a frame
    hit_z: float = 3.0
    canopy_height: float = 2.0       # raised surfaces above this hide the ground ...
    roof_roughness: float = 0.08     # ... a smooth one is a roof (not searchable), a rough one canopy
    canopy_seethrough: float = 0.5   # share of the ground visible through canopy gaps up close ...
    canopy_range: float = 10.0       # ... halving at this range (finer resolution resolves more gaps)
    quality_range: float = 25.0      # m: a view's quality halves at this range

    subsurface_prior: float = 0.01
    depression_depth: float = 0.3    # m below the ground plane counts as a depression
    depression: float = 0.8          # subsurface log-odds per frame for a clearly seen depression

    clamp: float = 8.0
    confirm_p: float = 0.9           # detections above this probability


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


class BeliefMap:
    def __init__(self, grid: GridMap, rig: StereoRig, image_size: tuple[int, int],
                 config: BeliefConfig | None = None):
        self.grid, self.rig, self.image_size = grid, rig, tuple(image_size)
        self.cfg = config or BeliefConfig()
        self._origin, self._shape = None, (0, 0)
        self._sync()

    # -- evidence --------------------------------------------------------------

    def observe(self, frame, t_world_cam: np.ndarray, pose_cov: np.ndarray | None = None) -> bool:
        """Fold one perception frame into the beliefs (call alongside GridMap.update)."""
        p = self.grid.project(frame, t_world_cam, pose_cov)
        if p is None:
            return False
        self._sync()
        cfg = self.cfg
        n = self._lo.size
        q = self._view_quality(p.depth)
        raised = p.height_ok & (p.height > cfg.canopy_height)
        smooth = np.isfinite(p.roughness) & (p.roughness < cfg.roof_roughness)
        roof, canopy = raised & smooth, raised & ~smooth
        ground_q = np.where(roof, 0.0, np.where(canopy, self._canopy_quality(p.depth), q))

        # Evidence comes from improving on the best look so far: repeating a similar view
        # re-reveals the same gaps and adds nothing; a closer look does.
        cell_q = np.zeros(n)
        np.maximum.at(cell_q, p.idx, ground_q)
        gain = np.maximum(cell_q - self._quality, 0.0)
        hit = np.zeros(n, bool)
        strong = np.isfinite(p.novelty) & (p.novelty >= cfg.hit_z)
        # Something unusual glimpsed under canopy is on the ground: place it where the ray meets
        # the ground, not on the canopy top (they're metres apart for slanted rays).
        hit[np.where(canopy, p.idx_ground, p.idx)[strong]] = True
        for votes, where in ((self._views, np.ones(len(p.idx), bool)), (self._roof_votes, roof),
                             (self._canopy_votes, canopy)):
            hit_cells = np.zeros(n, bool)  # once per cell per frame, however many samples land in it
            hit_cells[p.idx[where]] = True
            votes[hit_cells] += 1.0

        self._quality = np.maximum(self._quality, cell_q)
        self._lo += np.where(hit, cfg.hit, cfg.clear_view * gain)

        dep = p.height_ok & (p.height < -cfg.depression_depth)
        dep_q = np.zeros(n)
        np.maximum.at(dep_q, p.idx[dep], q[dep])
        self._ls += cfg.depression * dep_q
        np.clip(self._lo, -cfg.clamp, cfg.clamp, out=self._lo)
        np.clip(self._ls, -cfg.clamp, cfg.clamp, out=self._ls)
        return True

    # -- state ----------------------------------------------------------------

    @property
    def searchable(self) -> np.ndarray:
        """False where the cell has mostly been seen as a roof."""
        self._sync()
        frac = self._roof_votes / np.maximum(self._views, 1.0)
        return (frac < 0.5).reshape(self.grid.shape)

    @property
    def p_object(self) -> np.ndarray:
        self._sync()
        p = 1.0 / (1.0 + np.exp(-self._lo))
        return np.where(self.searchable.ravel(), p, 0.0).reshape(self.grid.shape)

    @property
    def p_subsurface(self) -> np.ndarray:
        self._sync()
        return (1.0 / (1.0 + np.exp(-self._ls))).reshape(self.grid.shape)

    @property
    def view_quality(self) -> np.ndarray:
        self._sync()
        return self._quality.reshape(self.grid.shape)

    def entropy(self, area: tuple[float, float, float, float] | None = None) -> dict:
        """Remaining uncertainty in bits: {'object', 'subsurface', 'total'}."""
        mask = self.grid.area_mask(*area) if area is not None else np.ones(self.grid.shape, bool)
        h_obj = float(_entropy(self.p_object)[mask & self.searchable].sum())
        h_sub = float(_entropy(self.p_subsurface)[mask].sum())
        return {"object": h_obj, "subsurface": h_sub, "total": h_obj + h_sub}

    def detections(self, kind: str = "object", min_cells: int = 2) -> list[dict]:
        """Confident hidden things as blobs: {"x", "y", "area_m2", "p"}, most probable first."""
        self._sync()
        if self._origin is None:
            return []
        prob = self.p_object if kind == "object" else self.p_subsurface
        n, labels = cv2.connectedComponents((prob >= self.cfg.confirm_p).astype(np.uint8), connectivity=8)
        out = []
        for k in range(1, n):
            rr, cc = np.nonzero(labels == k)
            if len(rr) < min_cells:
                continue
            w = prob[rr, cc]
            x, y = self.grid.cell_to_world(rr, cc)
            out.append({"x": float((x * w).sum() / w.sum()), "y": float((y * w).sum() / w.sum()),
                        "area_m2": len(rr) * self.grid.cfg.resolution ** 2, "p": float(w.max())})
        return sorted(out, key=lambda d: -d["p"])

    # -- reward -----------------------------------------------------------------

    def expected_gain(self, x: float, y: float, agl: float) -> float:
        """Approximate bits a nadir view from (x, y) at agl metres would remove. A view only helps
        where it beats the best look so far, so after a high pass over trees, a low pass is what still pays."""
        self._sync()
        if self._origin is None:
            return 0.0
        cfg = self.cfg
        hx = agl * self.image_size[0] / 2 / self.rig.fx
        hy = agl * self.image_size[1] / 2 / self.rig.fy
        cells, cx, cy = self._cells_near(x, y, math.hypot(hx, hy))
        # Range to the surface the camera would actually see (the canopy top over trees), as in observe().
        # The footprint on a raised surface is smaller than on the ground.
        top = np.maximum(np.nan_to_num(self.grid.height.ravel()[cells], nan=0.0), 0.0)
        depth = np.maximum(float(agl) - top, 0.5)
        shrink = depth / float(agl)
        inside = (np.abs(cx - x) <= hx * shrink) & (np.abs(cy - y) <= hy * shrink)
        cells, cx, cy, top, depth = cells[inside], cx[inside], cy[inside], top[inside], depth[inside]
        canopy = self._canopy_cells(cells)
        q_cell = np.where(canopy, self._canopy_quality(depth), self._view_quality(depth))
        q_cell = q_cell * self._line_of_sight(x, y, agl, cx, cy, top)  # occluded cells can't be seen from here
        improve = np.maximum(q_cell - self._quality[cells], 0.0)
        frac = 1.0 - np.exp(cfg.clear_view * improve)  # share of a cell's uncertainty the view resolves
        p_obj = self.p_object.ravel()[cells]
        return float((_entropy(p_obj) * self.searchable.ravel()[cells] * frac).sum())

    # -- internals -----------------------------------------------------------------

    def _line_of_sight(self, x, y, agl, cx, cy, top, samples: int = 12, margin: float = 0.3) -> np.ndarray:
        """1 where the straight line from the camera (x, y, agl) to each cell's surface clears every
        mapped height in between, else 0. Unmapped cells along the way count as ground."""
        t = np.linspace(0.05, 0.95, samples)[None, :]
        px = x + (cx[:, None] - x) * t
        py = y + (cy[:, None] - y) * t
        pz = agl + (top[:, None] - agl) * t
        r, c = self.grid.world_to_cell(px, py)
        ok = (r >= 0) & (r < self.grid.shape[0]) & (c >= 0) & (c < self.grid.shape[1])
        h = np.zeros(px.shape)
        h[ok] = np.nan_to_num(self.grid.height[r[ok], c[ok]], nan=0.0)
        return (~(h > pz + margin).any(axis=1)).astype(float)

    def _canopy_cells(self, cells: np.ndarray) -> np.ndarray:
        """Cells seen as canopy in most of their views."""
        return self._canopy_votes[cells] / np.maximum(self._views[cells], 1.0) > 0.5

    def _view_quality(self, depth) -> np.ndarray:
        return 1.0 / (1.0 + (np.asarray(depth) / self.cfg.quality_range) ** 2)

    def _canopy_quality(self, depth) -> np.ndarray:
        return self.cfg.canopy_seethrough / (1.0 + (np.asarray(depth) / self.cfg.canopy_range) ** 2)

    def _cells_near(self, x: float, y: float, radius: float):
        g = self.grid
        r0, c0 = g.world_to_cell(x - radius, y - radius)
        r1, c1 = g.world_to_cell(x + radius, y + radius)
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(r1) + 1, g.shape[0]), min(int(c1) + 1, g.shape[1])
        if r1 <= r0 or c1 <= c0:
            return np.zeros(0, int), np.zeros(0), np.zeros(0)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        cx, cy = g.cell_to_world(rr.ravel(), cc.ravel())
        near = (cx - x) ** 2 + (cy - y) ** 2 <= radius * radius
        return (rr.ravel() * g.shape[1] + cc.ravel())[near], cx[near], cy[near]

    def _sync(self) -> None:
        """Keep the belief layers on the same grid as the map, which grows as the drone flies."""
        g = self.grid
        if g.origin is None:
            return
        if self._origin is not None and g.shape == self._shape and np.allclose(g.origin, self._origin):
            return
        cfg = self.cfg
        new = {"_lo": _logit(cfg.object_prior), "_ls": _logit(cfg.subsurface_prior),
               "_quality": 0.0, "_roof_votes": 0.0, "_views": 0.0, "_canopy_votes": 0.0}
        fresh = {k: np.full(g.shape[0] * g.shape[1], v) for k, v in new.items()}
        if self._origin is not None:
            res = g.cfg.resolution
            left = int(round((self._origin[0] - g.origin[0]) / res))
            top = int(round((self._origin[1] - g.origin[1]) / res))
            h, w = self._shape
            for k in new:
                fresh[k].reshape(g.shape)[top:top + h, left:left + w] = getattr(self, k).reshape(h, w)
        for k, v in fresh.items():
            setattr(self, k, v)
        self._origin, self._shape = g.origin.copy(), g.shape
