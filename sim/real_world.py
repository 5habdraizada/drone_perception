"""Real places for the simulator: an orthophoto draped over a LiDAR surface model.

The area comes from tools/fetch_pdok.py. World coordinates are local: origin at the
area's centre, x east, y north, z up with 0 at the median ground level.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sim.synthetic_stereo import Scene


@dataclass
class RealArea:
    scene: Scene                 # what the simulator renders (surface model + orthophoto)
    truth_height: np.ndarray     # m above local ground, 0.5 m grid, row 0 at the south edge
    truth_res: float
    extent: float

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True height above ground at world points (nearest 0.5 m cell; NaN outside)."""
        c = np.floor((np.asarray(x) + self.extent / 2) / self.truth_res).astype(int)
        r = np.floor((np.asarray(y) + self.extent / 2) / self.truth_res).astype(int)
        n = self.truth_height.shape[0]
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        out = np.full(np.shape(r), np.nan)
        out[ok] = self.truth_height[r[ok], c[ok]]
        return out


def load_area(folder: str | Path) -> RealArea:
    folder = Path(folder)
    meta = json.loads((folder / "area.json").read_text())
    dsm = _read_dem(folder / "dsm.tif")
    dtm = _read_dem(folder / "dtm.tif")
    ground = _fill(dtm)                        # the ground model has holes under buildings
    surface = np.where(np.isnan(dsm), ground, dsm)  # surface holes are mostly water
    level = float(np.nanmedian(dtm))
    truth = np.clip(surface - ground, 0.0, None)

    ortho = cv2.imread(str(folder / "ortho.png"), cv2.IMREAD_GRAYSCALE)
    if ortho is None:
        raise FileNotFoundError(folder / "ortho.png")
    res = (meta["bbox"][2] - meta["bbox"][0]) / ortho.shape[0]
    # Rasters arrive north-up; the simulator wants row 0 at the south edge. Heights stay on
    # their native 0.5 m grid (small enough to stay in cache while ray marching).
    scene = Scene.from_arrays(np.flipud(surface - level), np.flipud(ortho.astype(np.float32) / 255.0), res)
    return RealArea(scene, np.flipud(truth).copy(), meta["dem_resolution"], meta["bbox"][2] - meta["bbox"][0])


def _read_dem(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    a = a.astype(np.float64)
    a[a > 1e30] = np.nan                       # AHN no-data
    return a


def _fill(a: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Fill NaN holes from their edges inwards with the average of known neighbours."""
    a = a.copy()
    k = np.ones((3, 3), np.float32)
    for _ in range(max_iter):
        hole = np.isnan(a)
        if not hole.any():
            break
        val = np.where(hole, 0.0, a).astype(np.float32)
        cnt = cv2.filter2D((~hole).astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        avg = cv2.filter2D(val, -1, k, borderType=cv2.BORDER_CONSTANT) / np.maximum(cnt, 1e-6)
        grow = hole & (cnt > 0)
        a[grow] = avg[grow]
    return a
