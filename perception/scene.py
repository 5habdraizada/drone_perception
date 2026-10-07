"""Scene understanding learned online: regions (online DP-means) and novelty (memory bank).

Each cell's feature is the backbone vector plus squashed height, slope and roughness.
Thresholds are relative to the frame's own feature noise. Cells without stereo depth
get a region from appearance alone and no novelty. Details in the README.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from perception.terrain import Terrain, box_sum


@dataclass
class SceneConfig:
    # Fused feature: appearance is unit length; geometry channels are tanh-squashed.
    geo_weight: float = 1.0        # scale of the geometry block; 1.0 makes a >= 1 m step alone a new region
    height_scale: float = 1.0      # m;   tanh(h / scale)
    slope_scale: float = 30.0      # deg; tanh(slope / scale)
    rough_scale: float = 0.05      # m;   tanh(roughness / scale)
    noise_rate: float = 0.1        # how fast the feature-noise estimate follows each frame

    # Regions
    new_region_factor: float = 2.0  # new region when farther than this × noise from every region
    merge_factor: float = 1.0       # merge regions whose centres are closer than this × noise
    max_regions: int = 12
    max_new_per_frame: int = 2      # limits how fast noise can create regions
    min_region_cells: int = 3       # similar outlier cells needed to start a region
    centroid_rate: float = 0.1      # how fast region centres follow new evidence (per frame)
    weight_rate: float = 0.02       # how fast "how common is this region" adapts
    prune_weight: float = 0.002     # regions rarer than this and unseen for prune_after frames are dropped
    prune_after: int = 100
    smooth_regions: bool = True     # relabel isolated single cells to their neighbours' majority

    # Novelty
    bank_size: int = 2000           # cell features remembered
    bank_insert: float = 0.05       # fraction of each frame's cells added to the memory
    knn: int = 5                    # neighbours averaged for the raw score
    score_rate: float = 0.05        # running centre/spread of raw scores
    min_score_std: float = 0.25     # floor on that spread, as a fraction of the feature noise
    warmup_frames: int = 5          # novelty is NaN until the model has seen this many frames
    seed: int = 0


@dataclass
class Region:
    id: int
    centroid: np.ndarray
    weight: float           # long-run fraction of cells it explains
    last_seen: int = 0


@dataclass
class SceneFrame:
    t: float
    regions: np.ndarray      # gh × gw region id
    novelty: np.ndarray      # gh × gw z-score vs everything seen before; NaN = not assessed
    contrast: np.ndarray     # gh × gw distance to the neighbourhood's mean feature; NaN = not assessed
    features: np.ndarray     # gh × gw × (D + 3) fused features
    geometry_valid: np.ndarray  # gh × gw: stereo depth available for the cell
    noise: float             # current feature-noise estimate (units of the fused features)
    terrain: Terrain | None = None
    region_info: dict = field(default_factory=dict)  # id -> {"cells", "weight", "height"}


class SceneModel:
    def __init__(self, config: SceneConfig | None = None, backbone=None):
        self.cfg = config or SceneConfig()
        self.backbone = backbone
        self.regions: list[Region] = []
        self.frames = 0
        self.noise = math.nan
        self._next_id = 0
        self._bank: np.ndarray | None = None
        self._bank_n = 0
        self._score_centre = math.nan
        self._score_spread = math.nan
        self._rng = np.random.default_rng(self.cfg.seed)

    def process(self, t: float, image: np.ndarray, terrain: Terrain) -> SceneFrame:
        """Run the backbone on the rectified left image and update with the terrain."""
        if self.backbone is None:
            raise RuntimeError("SceneModel was created without a backbone; use update()")
        appearance = self.backbone(image).resample(terrain.cell_height.shape)
        frame = self.update(t, appearance, terrain.cell_height, terrain.slope_deg, terrain.roughness)
        frame.terrain = terrain
        return frame

    def update(self, t: float, appearance: np.ndarray, height: np.ndarray,
               slope: np.ndarray, roughness: np.ndarray) -> SceneFrame:
        """One frame from per-cell arrays (gh × gw × D appearance, gh × gw geometry with NaNs)."""
        cfg = self.cfg
        gh, gw, _ = appearance.shape
        app = appearance / (np.linalg.norm(appearance, axis=-1, keepdims=True) + 1e-9)
        geo = cfg.geo_weight * np.stack([np.tanh(height / cfg.height_scale),
                                         np.tanh(slope / cfg.slope_scale),
                                         np.tanh(roughness / cfg.rough_scale)], axis=-1)
        geo_ok = np.isfinite(geo).all(axis=-1)
        fused = np.concatenate([app, np.where(geo_ok[..., None], geo, 0.0)], axis=-1)
        x = fused.reshape(gh * gw, -1).astype(np.float64)
        ok = geo_ok.ravel()

        contrast = _contrast(fused, geo_ok)
        if np.isfinite(contrast).any():
            med = float(np.nanmedian(contrast))
            self.noise = med if math.isnan(self.noise) else (1 - cfg.noise_rate) * self.noise + cfg.noise_rate * med
        noise = self.noise if math.isfinite(self.noise) else 0.5

        novelty = self._novelty(x, ok, noise)
        self._update_regions(x, ok, noise)
        ids = np.array([r.id for r in self.regions])
        regions = ids[np.argmin(self._distances2(x, ok), axis=1)].reshape(gh, gw)
        if cfg.smooth_regions:
            regions = _majority(regions)
        self._remember(x, ok)

        weights = {r.id: r.weight for r in self.regions}
        info = {}
        for i in np.unique(regions):
            h = height[regions == i]
            info[int(i)] = {"cells": int((regions == i).sum()), "weight": weights[int(i)],
                            "height": float(np.nanmedian(h)) if np.isfinite(h).any() else math.nan}
        self.frames += 1
        return SceneFrame(t, regions, novelty.reshape(gh, gw), contrast, fused.astype(np.float32),
                          geo_ok, noise, region_info=info)

    def _update_regions(self, x: np.ndarray, ok: np.ndarray, noise: float) -> None:
        cfg = self.cfg
        lam = cfg.new_region_factor * noise
        learn = x[ok] if ok.any() else x
        if not self.regions:
            self._add_region(learn.mean(axis=0), 1.0)

        # Spawn regions for groups of cells that fit none (the first frame may add several).
        # A group needs min_region_cells similar outliers: noise produces scattered
        # outliers, a genuinely new kind of surface produces a cluster of them.
        budget = cfg.max_regions if self.frames == 0 else cfg.max_new_per_frame
        everyone = np.ones(len(learn), bool)
        for _ in range(budget):
            if len(self.regions) >= cfg.max_regions:
                break
            far = self._distances2(learn, everyone).min(axis=1)
            outliers = np.flatnonzero(far > lam ** 2)
            spawned = False
            for seed in outliers[np.argsort(-far[outliers])][:20]:
                near = (far > lam ** 2) & (((learn - learn[seed]) ** 2).sum(axis=1) < (lam / 2) ** 2)
                if near.sum() >= cfg.min_region_cells:
                    self._add_region(learn[near].mean(axis=0), 0.0)
                    spawned = True
                    break
            if not spawned:
                break

        # Move centres towards this frame's members; fresh regions jump straight there.
        assign = np.argmin(self._distances2(learn, np.ones(len(learn), bool)), axis=1)
        for k, r in enumerate(self.regions):
            members = assign == k
            frac = members.mean()
            fresh = r.weight == 0.0
            r.weight = frac if self.frames == 0 else (1 - cfg.weight_rate) * r.weight + cfg.weight_rate * frac
            if members.any():
                rate = 1.0 if (fresh or self.frames == 0) else cfg.centroid_rate
                r.centroid = (1 - rate) * r.centroid + rate * learn[members].mean(axis=0)
                r.last_seen = self.frames
        if self.frames == 0:  # drop seeds that ended up explaining nothing
            self.regions = [r for r in self.regions if r.weight > 0] or self.regions
        total = sum(r.weight for r in self.regions)
        for r in self.regions:
            r.weight = r.weight / total if total > 0 else 1 / len(self.regions)

        self._merge(cfg.merge_factor * noise)
        self.regions = [r for r in self.regions
                        if r.weight >= cfg.prune_weight or self.frames - r.last_seen < cfg.prune_after] or self.regions

    def _merge(self, dist: float) -> None:
        while len(self.regions) > 1:
            cents = np.array([r.centroid for r in self.regions])
            d = np.linalg.norm(cents[:, None] - cents[None], axis=-1) + np.eye(len(cents)) * 1e9
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= dist:
                return
            a, b = sorted((self.regions[i], self.regions[j]), key=lambda r: r.id)  # keep the older id
            w = a.weight + b.weight
            if w > 0:
                a.centroid = (a.weight * a.centroid + b.weight * b.centroid) / w
            a.weight, a.last_seen = w, max(a.last_seen, b.last_seen)
            self.regions.remove(b)

    def _add_region(self, centroid: np.ndarray, weight: float) -> None:
        self.regions.append(Region(self._next_id, centroid.copy(), weight, self.frames))
        self._next_id += 1

    def _distances2(self, x: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """Squared distances to every region centre; cells without geometry compare appearance only."""
        cents = np.array([r.centroid for r in self.regions])
        d2 = (x ** 2).sum(1)[:, None] + (cents ** 2).sum(1)[None] - 2 * x @ cents.T
        if (~ok).any():
            xa, ca = x[~ok, :-3], cents[:, :-3]
            d2[~ok] = (xa ** 2).sum(1)[:, None] + (ca ** 2).sum(1)[None] - 2 * xa @ ca.T
        return np.maximum(d2, 0.0)

    def _novelty(self, x: np.ndarray, ok: np.ndarray, noise: float) -> np.ndarray:
        cfg = self.cfg
        out = np.full(len(x), np.nan)
        if self._bank is None or self._bank_n < cfg.knn or not ok.any():
            return out
        bank = self._bank[: self._bank_n]
        q = x[ok]
        d2 = (q ** 2).sum(1)[:, None] + (bank ** 2).sum(1)[None] - 2 * q @ bank.T
        nearest = np.sqrt(np.maximum(np.partition(d2, cfg.knn - 1, axis=1)[:, : cfg.knn], 0.0))
        raw = nearest.mean(axis=1)

        # Robust centre/spread of this frame's scores; most cells are ordinary.
        med = float(np.median(raw))
        spread = float(np.median(np.abs(raw - med))) * 1.4826
        if math.isnan(self._score_centre):
            self._score_centre, self._score_spread = med, spread
        if self.frames >= cfg.warmup_frames:
            scale = max(self._score_spread, cfg.min_score_std * noise, 1e-9)
            out[ok] = (raw - self._score_centre) / scale
        rate = max(cfg.score_rate, 1.0 / (self.frames + 1))  # plain average until it has history
        self._score_centre = (1 - rate) * self._score_centre + rate * med
        self._score_spread = (1 - rate) * self._score_spread + rate * spread
        return out

    def _remember(self, x: np.ndarray, ok: np.ndarray) -> None:
        """Add a random sample of this frame's cells to the bank (random replacement once full)."""
        cfg = self.cfg
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            return
        if self._bank is None:
            self._bank = np.empty((cfg.bank_size, x.shape[1]))
        n = max(1, int(round(cfg.bank_insert * cand.size)))
        if self.frames == 0:
            n = cand.size  # the first frame defines "normal" to start with
        pick = self._rng.choice(cand, size=min(n, cand.size), replace=False)
        for i in pick:
            if self._bank_n < cfg.bank_size:
                self._bank[self._bank_n] = x[i]
                self._bank_n += 1
            else:
                self._bank[self._rng.integers(cfg.bank_size)] = x[i]


def _contrast(fused: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Distance from each cell to the mean of its valid 8 neighbours."""
    w = valid.astype(np.float64)
    f = np.where(valid[..., None], fused, 0.0)
    nsum = box_sum(f * 1.0, 3) - f
    ncount = box_sum(w, 3) - w
    mean = nsum / np.maximum(ncount, 1)[..., None]
    out = np.linalg.norm(fused - mean, axis=-1)
    out[(ncount == 0) | ~valid] = np.nan
    return out.astype(np.float32)


def _majority(labels: np.ndarray) -> np.ndarray:
    """Relabel isolated cells (no 8-neighbour shares their label) to their neighbours' majority.

    Anything two cells or larger is kept, so small objects survive.
    """
    ids = np.unique(labels)
    votes = np.stack([box_sum((labels == i).astype(np.float64), 3) for i in ids], axis=-1)
    own = labels[..., None] == ids
    isolated = (votes * own).sum(axis=-1) <= 1  # only the cell itself
    neighbours = np.where(own, votes - 1, votes)
    return np.where(isolated, ids[np.argmax(neighbours, axis=-1)], labels)
