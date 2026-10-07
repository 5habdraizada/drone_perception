"""Terrain from a rectified, downward-facing stereo pair: ground plane, altitude, heights, slope.

Camera frame: x right, y down, z forward (OpenCV). Ground plane (nx, ny, nz, d):
n . X + d = 0 with |n| = 1 and d > 0, so d is the camera's altitude and
n . X + d is the height of a point above the ground.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoRig:
    """Intrinsics shared by both cameras after rectification, plus the baseline."""
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float  # metres

    @classmethod
    def from_Q(cls, Q: np.ndarray) -> "StereoRig":
        """Build from the 4x4 reprojection matrix returned by cv2.stereoRectify."""
        f = float(Q[2, 3])
        return cls(fx=f, fy=f, cx=float(-Q[0, 3]), cy=float(-Q[1, 3]),
                   baseline=abs(1.0 / float(Q[3, 2])))


@dataclass
class TerrainConfig:
    """Pixel-sized defaults are tuned for 320 px wide images; use for_width() otherwise."""

    # Stereo matching. "sgbm" is denser but pixel-locks, flattening slopes (see README).
    matcher: str = "bm"
    min_depth: float = 1.5          # m; sets the disparity search range when num_disparities is None
    max_depth: float = 40.0         # m; disparities implying anything farther are dropped
    num_disparities: int | None = None  # multiple of 16
    block_size: int = 15            # odd; StereoBM window. Larger = smoother, blurs edges
    sgbm_block_size: int = 5        # odd; StereoSGBM window (its smoothness term does the rest)
    texture_threshold: int = 10     # StereoBM only
    uniqueness_ratio: int = 10
    speckle_window: int = 100       # px area; smaller disconnected blobs are removed
    speckle_range_px: float = 2.0   # max disparity variation inside one blob

    # Ground plane
    ransac_iters: int = 200
    ransac_samples: int = 4000      # valid pixels subsampled for RANSAC
    inlier_px: float = 0.5          # disparity residual counted as "on the ground"
    max_tilt_deg: float = 40.0      # reject planes tilted more than this from the optical axis
    edge_window: int = 9            # px; neighbourhood checked for depth steps
    edge_px: float = 0.75           # disparity range within it that marks a step (excluded from the fit)
    prior_weight: float = 0.003     # with a prior: pull towards it, per inlier (ridge strength)
    prior_max_jump_m: float = 0.6   # with a prior: max altitude change between frames
    prior_max_angle_deg: float = 10.0  # with a prior: max ground-normal change between frames

    # Cell statistics
    cell_px: int = 16
    min_cell_valid: float = 0.4     # fraction of valid pixels needed for a cell to be reported
    slope_window: int = 3           # cells (odd); neighbourhood pooled for slope and roughness

    seed: int = 0

    @classmethod
    def for_width(cls, width: int, **overrides) -> "TerrainConfig":
        """Defaults with pixel-sized settings scaled from 320 px to `width` (same grid of cells)."""
        s = width / 320
        odd = lambda v: max(3, int(round(v)) | 1)
        scaled = dict(block_size=odd(15 * s), edge_window=odd(9 * s),
                      cell_px=max(4, int(round(16 * s))), speckle_window=int(100 * s * s))
        scaled.update(overrides)
        return cls(**scaled)


@dataclass
class Terrain:
    """Geometry of one stereo frame. Invalid values are NaN."""
    disparity: np.ndarray       # H×W px
    points: np.ndarray          # H×W×3, camera frame, m
    height: np.ndarray          # H×W, m above the ground plane
    plane: np.ndarray | None    # (nx, ny, nz, d) in the camera frame
    altitude: float             # m, camera height above the plane
    tilt_deg: float             # angle between optical axis and plane normal
    cell_height: np.ndarray     # gh×gw median height, m
    slope_deg: np.ndarray       # gh×gw local surface angle relative to the ground plane
    roughness: np.ndarray       # gh×gw RMS deviation from a local plane, m
    valid_fraction: float       # fraction of pixels with a disparity
    inlier_fraction: float      # fraction of valid pixels on the ground plane


class TerrainEstimator:
    def __init__(self, rig: StereoRig, config: TerrainConfig | None = None):
        self.rig = rig
        self.cfg = cfg = config or TerrainConfig()
        nd = cfg.num_disparities or 16 * math.ceil(rig.fx * rig.baseline / cfg.min_depth / 16)
        self.num_disparities = max(16, nd)
        self._matcher = self._make_matcher()
        self._rng = np.random.default_rng(cfg.seed)
        self._rays: tuple[np.ndarray, np.ndarray] | None = None

    def process(self, left: np.ndarray, right: np.ndarray,
                plane_prior: np.ndarray | None = None) -> Terrain:
        """plane_prior: the previous frame's plane (or one predicted from the pose). Only ground
        planes close to it are accepted, so a roof filling the view isn't taken for the ground."""
        disp = self.disparity(left, right)
        xr, yr = self._normalised_coords(disp.shape)
        valid = np.isfinite(disp)
        # Stereo matching smears depth steps into ramps; keep them out of the ground fit.
        fit = valid & ~self.depth_edges(disp)

        plane, inlier_frac = self.fit_ground_plane(xr[fit], yr[fit], disp[fit], plane_prior)
        points = self.points_from_disparity(disp)

        if plane is None:
            nan = np.full(disp.shape, np.nan, np.float32)
            gh, gw = disp.shape[0] // self.cfg.cell_px, disp.shape[1] // self.cfg.cell_px
            cells = np.full((gh, gw), np.nan, np.float32)
            return Terrain(disp, points, nan, None, math.nan, math.nan,
                           cells, cells.copy(), cells.copy(), float(valid.mean()), 0.0)

        n, d = plane[:3], plane[3]
        height = (points @ n + d).astype(np.float32)
        cell_height, slope, rough = self._cell_stats(points, height, n)
        tilt = math.degrees(math.acos(float(np.clip(-n[2], -1.0, 1.0))))
        return Terrain(disp, points, height, plane, float(d), tilt,
                       cell_height, slope, rough, float(valid.mean()), inlier_frac)

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raw = self._matcher.compute(_gray(left), _gray(right)).astype(np.float32) / 16.0
        d_min = self.rig.fx * self.rig.baseline / self.cfg.max_depth
        raw[raw <= max(d_min, 1e-3)] = np.nan
        return raw

    def depth_edges(self, disp: np.ndarray) -> np.ndarray:
        """Pixels near a depth step or a hole: local disparity range above edge_px."""
        filled = np.nan_to_num(disp, nan=0.0)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.edge_window,) * 2)
        return cv2.dilate(filled, k) - cv2.erode(filled, k) > self.cfg.edge_px

    def points_from_disparity(self, disp: np.ndarray) -> np.ndarray:
        xr, yr = self._normalised_coords(disp.shape)
        z = self.rig.fx * self.rig.baseline / disp
        return np.stack([xr * z, yr * z, z], axis=-1).astype(np.float32)

    def fit_ground_plane(self, xr: np.ndarray, yr: np.ndarray, d: np.ndarray,
                         prior: np.ndarray | None = None) -> tuple[np.ndarray | None, float]:
        """RANSAC fit of d = a*xr + b*yr + c, returned as a metric plane (n, d)."""
        cfg = self.cfg
        if d.size < 50:
            return None, 0.0
        obs = np.stack([xr, yr, np.ones_like(xr)], axis=1).astype(np.float64)
        d = d.astype(np.float64)
        if d.size > cfg.ransac_samples:
            pick = self._rng.choice(d.size, cfg.ransac_samples, replace=False)
            s_obs, s_d = obs[pick], d[pick]
        else:
            s_obs, s_d = obs, d

        best = self._ransac(s_obs, s_d, prior)
        if best is None:
            return None, 0.0

        # Least-squares refit on the inliers. With a prior, pull gently towards it: when only a
        # strip of ground is visible, its tilt is barely observable.
        prior_cf = self._plane_to_coeffs(prior) if prior is not None else None
        for _ in range(2):
            mask = np.abs(s_obs @ best - s_d) < cfg.inlier_px
            if mask.sum() < 3:
                return None, 0.0
            a_mat, b_vec = s_obs[mask], s_d[mask]
            if prior_cf is not None:
                lam = math.sqrt(cfg.prior_weight * mask.sum())
                a_mat = np.vstack([a_mat, lam * np.eye(3)])
                b_vec = np.concatenate([b_vec, lam * prior_cf])
            best = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]

        inlier_frac = float(np.mean(np.abs(obs @ best - d) < cfg.inlier_px))
        return self._coeffs_to_plane(best), inlier_frac

    def _make_matcher(self):
        cfg = self.cfg
        if cfg.matcher == "bm":
            m = cv2.StereoBM_create(numDisparities=self.num_disparities, blockSize=cfg.block_size)
            m.setTextureThreshold(cfg.texture_threshold)
            m.setUniquenessRatio(cfg.uniqueness_ratio)
            m.setSpeckleWindowSize(cfg.speckle_window)
            m.setSpeckleRange(int(round(16 * cfg.speckle_range_px)))  # BM takes 1/16 px units
            return m
        if cfg.matcher == "sgbm":
            bs = cfg.sgbm_block_size
            return cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disparities, blockSize=bs,
                P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
                uniquenessRatio=cfg.uniqueness_ratio, speckleWindowSize=cfg.speckle_window,
                speckleRange=max(1, int(round(cfg.speckle_range_px))),  # SGBM scales by 16 itself
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        raise ValueError(f"unknown matcher {cfg.matcher!r}; use 'bm' or 'sgbm'")

    def _ransac(self, s_obs: np.ndarray, s_d: np.ndarray, prior: np.ndarray | None) -> np.ndarray | None:
        cfg = self.cfg
        idx = self._rng.integers(0, s_d.size, size=(cfg.ransac_iters, 3))  # 3-point hypotheses, batched
        a_mat, b_vec = s_obs[idx], s_d[idx]
        ok = np.abs(np.linalg.det(a_mat)) > 1e-9
        coeffs = np.linalg.solve(a_mat[ok], b_vec[ok][..., None])[..., 0]
        if prior is not None:
            coeffs = np.vstack([coeffs, self._plane_to_coeffs(prior)])

        normals = -coeffs / np.linalg.norm(coeffs, axis=1, keepdims=True)
        coeffs = coeffs[-normals[:, 2] >= math.cos(math.radians(cfg.max_tilt_deg))]
        if prior is not None and len(coeffs):
            # Gate on the previous ground; if nothing passes (ground hidden), fall back to ungated.
            planes = np.array([self._coeffs_to_plane(cf) for cf in coeffs])
            near = (np.abs(planes[:, 3] - prior[3]) <= cfg.prior_max_jump_m) & (
                planes[:, :3] @ np.asarray(prior[:3]) >= math.cos(math.radians(cfg.prior_max_angle_deg)))
            if near.any():
                coeffs = coeffs[near]
        if len(coeffs) == 0:
            return None

        inliers = np.abs(s_obs @ coeffs.T - s_d[:, None]) < cfg.inlier_px
        return coeffs[np.argmax(inliers.sum(axis=0))]

    def _coeffs_to_plane(self, coeffs: np.ndarray) -> np.ndarray:
        # d_px = -(f*B / D) * (n . ray)  =>  coeffs = -(f*B / D) * n
        scale = np.linalg.norm(coeffs)
        n = -coeffs / scale
        return np.append(n, self.rig.fx * self.rig.baseline / scale)

    def _plane_to_coeffs(self, plane: np.ndarray) -> np.ndarray:
        return -(self.rig.fx * self.rig.baseline / plane[3]) * np.asarray(plane[:3])

    def _normalised_coords(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._rays is None or self._rays[0].shape != shape:
            h, w = shape
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._rays = ((u - self.rig.cx) / self.rig.fx, (v - self.rig.cy) / self.rig.fy)
        return self._rays

    def _cell_stats(self, points: np.ndarray, height: np.ndarray, n_ground: np.ndarray):
        """Per-cell median height, plus slope and roughness from a PCA plane fit pooled over
        slope_window × slope_window cells (one cell alone is tilted by disparity quantisation)."""
        s = max(1, self.cfg.cell_px // 16)  # ~16 × 16 samples per cell is plenty
        points, height = points[::s, ::s], height[::s, ::s]
        c = self.cfg.cell_px // s
        gh, gw = height.shape[0] // c, height.shape[1] // c
        pts = points[: gh * c, : gw * c]
        hts = height[: gh * c, : gw * c].reshape(gh, c, gw, c).transpose(0, 2, 1, 3).reshape(gh, gw, c * c)

        valid = np.isfinite(pts[..., 2])

        origin = pts[valid].mean(axis=0) if valid.any() else np.zeros(3, np.float32)
        p = np.where(valid[..., None], pts - origin, 0.0).astype(np.float32)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        chans = np.stack([valid.astype(np.float32), x, y, z,
                          x * x, x * y, x * z, y * y, y * z, z * z], axis=-1)
        sums = _block_sum(chans, c).astype(np.float64)
        count = np.rint(sums[..., 0]).astype(np.int64)
        enough = count >= max(3, self.cfg.min_cell_valid * c * c)
        s1 = sums[..., 1:4]
        s2 = sums[..., [4, 5, 6, 5, 7, 8, 6, 8, 9]].reshape(gh, gw, 3, 3)
        n_w = box_sum(count.astype(np.float64), self.cfg.slope_window)
        s1_w = box_sum(s1, self.cfg.slope_window)
        s2_w = box_sum(s2, self.cfg.slope_window)

        n_safe = np.maximum(n_w, 1.0)
        mean = s1_w / n_safe[..., None]
        cov = s2_w / n_safe[..., None, None] - mean[..., :, None] * mean[..., None, :]
        cov[~enough] = np.eye(3)  # placeholder so eigh stays well-defined
        evals, evecs = np.linalg.eigh(cov)

        local_n = evecs[..., :, 0]  # eigenvector of the smallest eigenvalue
        slope = np.degrees(np.arccos(np.clip(np.abs(local_n @ n_ground), 0.0, 1.0)))
        rough = np.sqrt(np.clip(evals[..., 0], 0.0, None))

        # Median of valid heights: NaNs sorted to the end.
        srt = np.sort(np.where(np.isfinite(hts), hts, np.inf), axis=-1)
        lo = np.take_along_axis(srt, (np.maximum(count - 1, 0) // 2)[..., None], axis=-1)[..., 0]
        hi = np.take_along_axis(srt, (count // 2)[..., None], axis=-1)[..., 0]
        med = np.where(count % 2 == 1, lo, 0.5 * (lo + hi))

        out = [med, slope, rough]
        for a in out:
            a[~enough] = np.nan
        return tuple(a.astype(np.float32) for a in out)


def _block_sum(a: np.ndarray, c: int) -> np.ndarray:
    """Sum non-overlapping c×c blocks (INTER_AREA at an integer factor is an exact, fast block mean)."""
    h, w = a.shape[:2]
    out = cv2.resize(a, (w // c, h // c), interpolation=cv2.INTER_AREA)
    return out.reshape(h // c, w // c, *a.shape[2:]) * float(c * c)


def box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over a k×k neighbourhood of the first two axes, zero-padded at the edges."""
    if k <= 1:
        return a
    r = k // 2
    padded = np.pad(a, [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2))
    gh, gw = a.shape[:2]
    out = np.zeros_like(a)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy + gh, dx:dx + gw]
    return out


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img, dtype=np.uint8)
