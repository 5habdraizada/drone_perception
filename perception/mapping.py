"""Context map: a world-fixed 2D grid of height, elevation, regions and anomaly evidence.

update() takes the rectified left camera's 4 × 4 world-from-camera pose (OpenCV camera
axes, world z up). The grid grows as the drone flies past its edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

import cv2
import numpy as np

from perception.terrain import StereoRig


class PoseSource(Protocol):
    def get_pose(self, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """(4 × 4 world-from-body transform at time t, 6 × 6 covariance or None).

        Covariance order: x, y, z (m), roll, pitch, yaw (rad). World z is up.
        """
        ...


def camera_from_body(t_world_body: np.ndarray, t_body_cam: np.ndarray) -> np.ndarray:
    """World-from-camera from world-from-body and the camera's mounting (body-from-camera)."""
    return t_world_body @ t_body_cam


def pose_from_quaternion(x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """4 × 4 world-from-camera transform from a position and a (w, x, y, z) unit quaternion."""
    q = np.array([qw, qx, qy, qz], np.float64)
    w, a, b, c = q / np.linalg.norm(q)
    t = np.eye(4)
    t[:3, :3] = [[1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
                 [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
                 [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)]]
    t[:3, 3] = (x, y, z)
    return t


def quaternion_from_rotation(r: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) unit quaternion of a 3 × 3 rotation matrix."""
    m = np.asarray(r, np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        q = (s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4)
    return tuple(float(v) for v in q)


def rectified_camera(t_world_cam: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """World pose of the rectified frame, given the raw left camera's pose and Rectifier.R1."""
    t = np.eye(4)
    t[:3, :3] = np.asarray(r1).T  # R1 maps raw-camera coordinates into the rectified frame
    return t_world_cam @ t


@dataclass
class MapConfig:
    resolution: float = 0.25        # m per map cell
    initial_size: float = 40.0      # m, square around the first observation; grows as needed
    samples_per_cell: int = 2       # projected sub-points per scene cell, per axis (fills gaps when high)
    depth_ref: float = 10.0         # m; an observation from this far counts half as much as one from close by
    pose_sigma_ref: float = 0.5     # m; pose uncertainty at which observations count half

    # Anomaly evidence
    hit_z: float = 3.0              # novelty z-score that counts as a hit
    hit_radius: int = 1             # map cells around a hit that also get it (pose jitter, cell edges)
    min_hits: int = 2               # separate frames with a hit needed to confirm an anomaly
    max_region_slots: int = 24      # distinct region ids the map keeps votes for


class GridMap:
    def __init__(self, rig: StereoRig, image_size: tuple[int, int], config: MapConfig | None = None):
        """rig/image_size: the rectified camera the scene frames come from (e.g. Rectifier.rig, out_size)."""
        self.rig, self.image_size = rig, tuple(image_size)
        self.cfg = config or MapConfig()
        self.origin: np.ndarray | None = None  # world (x, y) of cell (0, 0)'s corner
        self.shape = (0, 0)
        self._slots: dict[int, int] = {}
        self.frames = 0

    def update(self, frame, t_world_cam: np.ndarray, pose_cov: np.ndarray | None = None) -> bool:
        """Add one SceneFrame (needs frame.terrain). Returns False if it couldn't be placed."""
        p = self.project(frame, t_world_cam, pose_cov)
        if p is None:
            return False
        cfg = self.cfg
        idx, w, h, hk, xy = p.idx, p.weight, p.height, p.height_ok, p.xyz

        # Cells without a measured height are placed as if on the ground, which is fine for
        # regions and coverage but would corrupt the height and elevation layers.
        np.add.at(self._w, idx, w)
        np.add.at(self._hw, idx[hk], w[hk])
        np.add.at(self._hsum, idx[hk], (w * h)[hk])
        np.add.at(self._zsum, idx[hk], (w * xy[:, 2])[hk])

        # At most one hit per map cell per frame, spread to its neighbours.
        nv = p.novelty
        nk = np.isfinite(nv)
        if nk.any():
            np.maximum.at(self._peak, idx[nk], nv[nk])
            strong = idx[nk & (nv >= cfg.hit_z)]
            if strong.size:
                hit = np.zeros(self._w.size, np.uint8)
                hit[strong] = 1
                if cfg.hit_radius > 0:
                    k = 2 * cfg.hit_radius + 1
                    hit = cv2.dilate(hit.reshape(self.shape), np.ones((k, k), np.uint8)).ravel()
                self._hits += hit

        for rid in np.unique(p.regions):
            slot = self._slot(int(rid))
            if slot is not None:
                m = p.regions == rid
                np.add.at(self._votes[:, slot], idx[m], w[m])

        seen = np.unique(idx)
        self._seen[seen] += 1
        self._last[seen] = frame.t
        self.frames += 1
        return True

    def project(self, frame, t_world_cam: np.ndarray, pose_cov: np.ndarray | None = None):
        """Where each scene cell's sub-points land in the map, with what they observed.

        Returns None if the frame has no ground plane; otherwise a namespace of per-sample arrays:
        idx (flat map cell of the observed surface), idx_ground (where the same ray meets the ground),
        weight, height (m above ground) and height_ok, novelty, regions, roughness (NaN if the frame
        has none), depth (m along the optical axis), xyz (world).
        Grows the grid to fit. Shared by update() and the belief map.
        """
        terrain = frame.terrain
        if terrain is None or terrain.plane is None:
            return None
        cfg = self.cfg
        gh, gw = frame.regions.shape
        n, d = np.asarray(terrain.plane[:3]), float(terrain.plane[3])

        # Sub-points of every scene cell, as rays in the rectified camera. Dense enough that they
        # land closer together than map cells (a scene cell spans altitude * cell_px / f metres),
        # so no map cell under the camera is skipped.
        cw, ch = self.image_size[0] / gw, self.image_size[1] / gh
        span = max(d, 1e-3) * max(cw / self.rig.fx, ch / self.rig.fy)
        s = int(np.clip(math.ceil(span / (0.7 * cfg.resolution)), cfg.samples_per_cell, 12))
        sub = (np.arange(s) + 0.5) / s
        u = ((np.arange(gw)[:, None] + sub[None]) * cw).ravel()
        v = ((np.arange(gh)[:, None] + sub[None]) * ch).ravel()
        uu, vv = np.meshgrid(u, v)                                  # (gh*s) × (gw*s)
        rays = np.stack([(uu - self.rig.cx) / self.rig.fx, (vv - self.rig.cy) / self.rig.fy,
                         np.ones_like(uu)], axis=-1)
        cell = lambda a: np.repeat(np.repeat(a, s, axis=0), s, axis=1)

        height = cell(terrain.cell_height)
        h_ok = np.isfinite(height)
        # Point on the surface along each ray: n·(t r) + d = h  ->  t = (h - d) / (n·r).
        denom = rays @ n
        depth = (np.where(h_ok, height, 0.0) - d) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        good = np.isfinite(depth) & (depth > 0)
        pts_cam = rays * depth[..., None]
        pts = pts_cam @ t_world_cam[:3, :3].T + t_world_cam[:3, 3]

        weight = 1.0 / (1.0 + (depth / cfg.depth_ref) ** 2)
        if pose_cov is not None:
            sigma = math.sqrt(max(float(np.trace(np.asarray(pose_cov)[:2, :2])) / 2, 0.0))
            weight = weight / (1.0 + (sigma / cfg.pose_sigma_ref) ** 2)

        rough = getattr(terrain, "roughness", None)
        sel = good.ravel()
        if not sel.any():
            return None  # nothing in front of the camera (e.g. flying inside a canopy)
        xyz = pts.reshape(-1, 3)[sel]
        # Where the same rays meet the ground: what a glimpse through a gap in a canopy actually shows.
        depth0 = -d / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        ground = ((rays * depth0[..., None]) @ t_world_cam[:3, :3].T + t_world_cam[:3, 3]).reshape(-1, 3)[sel]
        ground_ok = np.isfinite(ground[:, 0]) & (depth0.ravel()[sel] > 0)
        ground[~ground_ok] = xyz[~ground_ok]
        self._ensure(np.concatenate([xyz[:, 0], ground[:, 0]]), np.concatenate([xyz[:, 1], ground[:, 1]]))
        return SimpleNamespace(
            idx=self._flat_index(xyz[:, 0], xyz[:, 1]), idx_ground=self._flat_index(ground[:, 0], ground[:, 1]),
            weight=weight.ravel()[sel],
            height=height.ravel()[sel], height_ok=h_ok.ravel()[sel],
            novelty=cell(frame.novelty).ravel()[sel], regions=cell(frame.regions).ravel()[sel],
            roughness=(cell(rough).ravel()[sel] if rough is not None else np.full(int(sel.sum()), np.nan)),
            depth=depth.ravel()[sel], xyz=xyz)

    @property
    def height(self) -> np.ndarray:
        return self._layer(self._hsum, self._hw)

    @property
    def elevation(self) -> np.ndarray:
        return self._layer(self._zsum, self._hw)

    @property
    def anomaly_hits(self) -> np.ndarray:
        return self._hits.reshape(self.shape)

    @property
    def novelty_peak(self) -> np.ndarray:
        return self._peak.reshape(self.shape)

    @property
    def seen(self) -> np.ndarray:
        return self._seen.reshape(self.shape)

    @property
    def last_seen(self) -> np.ndarray:
        return self._last.reshape(self.shape)

    @property
    def regions(self) -> np.ndarray:
        """Region id seen most at each map cell; -1 where nothing was seen."""
        if not self._slots:
            return np.full(self.shape, -1)
        ids = np.empty(len(self._slots), int)  # slots are handed out as 0, 1, 2, ...
        for rid, slot in self._slots.items():
            ids[slot] = rid
        votes = self._votes[:, : len(ids)]
        out = np.where(votes.sum(axis=1) > 0, ids[np.argmax(votes, axis=1)], -1)
        return out.reshape(self.shape)

    def ensure_area(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        """Make sure the grid covers this rectangle (e.g. a search area), so it can be planned over."""
        self._ensure(np.array([xmin, xmax]), np.array([ymin, ymax]))

    def area_mask(self, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
        """Boolean mask of map cells whose centres lie inside the rectangle."""
        rr, cc = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        x, y = self.cell_to_world(rr, cc)
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def world_to_cell(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) of world points; row grows with y."""
        r = np.floor((np.asarray(y) - self.origin[1]) / self.cfg.resolution).astype(int)
        c = np.floor((np.asarray(x) - self.origin[0]) / self.cfg.resolution).astype(int)
        return r, c

    def cell_to_world(self, r, c) -> tuple[np.ndarray, np.ndarray]:
        """World (x, y) of cell centres."""
        res = self.cfg.resolution
        return self.origin[0] + (np.asarray(c) + 0.5) * res, self.origin[1] + (np.asarray(r) + 0.5) * res

    def anomalies(self, min_hits: int | None = None, min_cells: int = 2) -> list[dict]:
        """Confirmed unusual spots as blobs, strongest first.

        Each: {"x", "y" (world centre, m), "area_m2", "peak_novelty",
               "height" (90th percentile: the blob includes some ground around the object, m),
               "hits" (most frames with a hit), "region" (most common region id)}.
        """
        thr = self.cfg.min_hits if min_hits is None else min_hits
        hits = self.anomaly_hits
        mask = (hits >= thr).astype(np.uint8)
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        height, peak, regions = self.height, self.novelty_peak, self.regions
        out = []
        for k in range(1, n):
            rr, cc = np.nonzero(labels == k)
            if len(rr) < min_cells:
                continue
            x, y = self.cell_to_world(rr, cc)
            hv = height[rr, cc]
            ids, counts = np.unique(regions[rr, cc], return_counts=True)
            out.append({"x": float(x.mean()), "y": float(y.mean()),
                        "area_m2": len(rr) * self.cfg.resolution ** 2,
                        "peak_novelty": float(np.nanmax(peak[rr, cc])),
                        "height": float(np.nanpercentile(hv, 90)) if np.isfinite(hv).any() else math.nan,
                        "hits": int(hits[rr, cc].max()),
                        "region": int(ids[np.argmax(counts)])})
        return sorted(out, key=lambda a: -a["peak_novelty"])

    def frontier(self) -> np.ndarray:
        """Boolean mask of unseen cells touching seen ones: where exploring would add map."""
        seen = (self.seen > 0).astype(np.uint8)
        grown = cv2.dilate(seen, np.ones((3, 3), np.uint8))
        return (grown > 0) & (seen == 0)

    def clearance(self, x: float, y: float, radius: float) -> float:
        """Highest mapped surface (m above ground) within `radius` of (x, y); NaN if none mapped."""
        r0, c0 = self.world_to_cell(x, y)
        k = int(math.ceil(radius / self.cfg.resolution))
        h = self.height
        rs = slice(min(max(r0 - k, 0), self.shape[0]), min(max(r0 + k + 1, 0), self.shape[0]))
        cs = slice(min(max(c0 - k, 0), self.shape[1]), min(max(c0 + k + 1, 0), self.shape[1]))
        patch = h[rs, cs]
        rr, cc = np.mgrid[rs, cs]
        xw, yw = self.cell_to_world(rr, cc)
        inside = (xw - x) ** 2 + (yw - y) ** 2 <= radius ** 2
        vals = patch[inside & np.isfinite(patch)] if patch.size else np.array([])
        return float(vals.max()) if vals.size else math.nan

    def render(self, height_range: tuple[float, float] = (-1.0, 2.5), crop: bool = True,
               scale: int = 1) -> np.ndarray:
        """BGR panel: height | regions | anomaly hits | times seen. North (+y) is up.

        crop: only the explored area (plus a small margin); scale: pixels per map cell.
        """
        def colour(a, lo, hi, cmap):
            norm = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)
            img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
            img[~np.isfinite(a)] = 30
            return img

        seen = self.seen > 0
        regions = self.regions
        rng = np.random.default_rng(0)
        palette = rng.integers(60, 255, (max(regions.max() + 1, 1), 3)).astype(np.uint8)
        reg = np.where(seen[..., None], palette[np.maximum(regions, 0)], 30).astype(np.uint8)
        hits = np.where(seen, self.anomaly_hits, np.nan)
        count = np.where(seen, np.log1p(self.seen), np.nan)
        panels = [colour(self.height, *height_range, cv2.COLORMAP_TURBO), reg,
                  colour(hits, 0.0, 2.0 * self.cfg.min_hits, cv2.COLORMAP_INFERNO),
                  colour(count, 0.0, max(float(np.nanmax(count)), 1e-6) if seen.any() else 1.0,
                         cv2.COLORMAP_VIRIDIS)]
        if crop and seen.any():
            rr, cc = np.nonzero(seen)
            m = 4
            rs = slice(max(rr.min() - m, 0), rr.max() + m + 1)
            cs = slice(max(cc.min() - m, 0), cc.max() + m + 1)
            panels = [p[rs, cs] for p in panels]
        panels = [cv2.resize(np.flipud(p), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                  for p in panels]
        gap = np.full((panels[0].shape[0], 4, 3), 255, np.uint8)
        return np.hstack([x for p in panels for x in (p, gap)][:-1])

    def _layer(self, total: np.ndarray, weight: np.ndarray) -> np.ndarray:
        out = np.full(weight.shape, np.nan)
        ok = weight > 0
        out[ok] = total[ok] / weight[ok]
        return out.reshape(self.shape)

    def _flat_index(self, x, y) -> np.ndarray:
        r, c = self.world_to_cell(x, y)
        return r * self.shape[1] + c

    def _ensure(self, x: np.ndarray, y: np.ndarray) -> None:
        """Allocate, or grow the grid so every (x, y) falls inside it."""
        res = self.cfg.resolution
        if self.origin is None:
            half = self.cfg.initial_size / 2
            self.origin = np.array([np.median(x) - half, np.median(y) - half])
            n = int(math.ceil(self.cfg.initial_size / res))
            self._allocate((n, n))
        r, c = self.world_to_cell(x, y)
        h, w = self.shape
        top, bottom = max(0, -int(r.min())), max(0, int(r.max()) - h + 1)
        left, right = max(0, -int(c.min())), max(0, int(c.max()) - w + 1)
        if top or bottom or left or right:
            # Grow by at least a quarter of the current size on each side that needs it.
            grow = lambda need, size: need and max(need, size // 4)
            top, bottom, left, right = grow(top, h), grow(bottom, h), grow(left, w), grow(right, w)
            self._pad(top, bottom, left, right)

    def _allocate(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        n = shape[0] * shape[1]
        self._w, self._hw, self._hsum, self._zsum = (np.zeros(n) for _ in range(4))
        self._hits, self._peak = np.zeros(n, np.int32), np.full(n, -np.inf)
        self._seen, self._last = np.zeros(n, np.int32), np.full(n, np.nan)
        self._votes = np.zeros((n, self.cfg.max_region_slots), np.float32)

    def _pad(self, top: int, bottom: int, left: int, right: int) -> None:
        h, w = self.shape
        new = (h + top + bottom, w + left + right)

        def pad(a, fill):
            a2 = a.reshape(h, w, *a.shape[1:])
            widths = [(top, bottom), (left, right)] + [(0, 0)] * (a2.ndim - 2)
            return np.pad(a2, widths, constant_values=fill).reshape(new[0] * new[1], *a.shape[1:])

        self._w, self._hw, self._hsum, self._zsum = (
            pad(a, 0.0) for a in (self._w, self._hw, self._hsum, self._zsum))
        self._peak, self._seen, self._hits = pad(self._peak, -np.inf), pad(self._seen, 0), pad(self._hits, 0)
        self._last, self._votes = pad(self._last, np.nan), pad(self._votes, 0.0)
        self.origin = self.origin - np.array([left, top]) * self.cfg.resolution
        self.shape = new

    def _slot(self, region_id: int) -> int | None:
        if region_id < 0:
            return None
        if region_id not in self._slots:
            if len(self._slots) >= self.cfg.max_region_slots:
                return None  # out of slots: this region's votes are dropped
            self._slots[region_id] = len(self._slots)
        return self._slots[region_id]
