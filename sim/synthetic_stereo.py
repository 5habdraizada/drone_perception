"""Synthetic downward-facing stereo scenes with exact ground truth.

The world is a heightfield over a flat ground (z = 0, z up) with rectangular
boxes (h > 0) and pits (h < 0). Both cameras are ideal rectified pinholes, so
the rendered pair can be fed straight into the terrain estimator.

Rendering is per-pixel ray marching against the heightfield, then bisection
for the exact hit, so walls, occlusions and parallax are all physically right.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.terrain import StereoRig


@dataclass(frozen=True)
class Block:
    """Axis-aligned block in world metres. h > 0 is a structure, h < 0 a pit."""
    x: float
    y: float
    w: float  # size along x
    l: float  # size along y
    h: float
    pattern: str = "noise"  # "noise" (blotchy, like the ground) or "stripes" (an unusual object)


@dataclass
class Scene:
    extent: float = 14.0      # metres, square centred on the origin
    res: float = 0.01         # metres per raster cell
    blocks: list[Block] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        n = int(round(self.extent / self.res))
        rng = np.random.default_rng(self.seed)
        self.height = np.zeros((n, n), np.float32)
        self.texture = _noise_texture(rng, n, scales=(2, 6, 20, 60))
        alt = _noise_texture(rng, n, scales=(3, 10, 30))
        for b in self.blocks:
            r0, r1 = self._idx(b.y - b.l / 2), self._idx(b.y + b.l / 2)
            c0, c1 = self._idx(b.x - b.w / 2), self._idx(b.x + b.w / 2)
            self.height[r0:r1, c0:c1] = b.h
            if b.pattern == "stripes":
                cols = np.arange(c0, c1) * self.res
                stripes = (np.sin(2 * np.pi * cols / 0.15) > 0).astype(np.float32)
                self.texture[r0:r1, c0:c1] = 0.15 + 0.7 * stripes[None, :]
            else:
                # Give structures their own look so appearance features can tell them apart.
                self.texture[r0:r1, c0:c1] = 0.5 * alt[r0:r1, c0:c1] + (0.35 if b.h > 0 else 0.1)
        self.texture = np.clip(self.texture, 0.0, 1.0)

    def _idx(self, coord: float) -> int:
        return int(round((coord + self.extent / 2) / self.res))

    def sample(self, x: np.ndarray, y: np.ndarray, field_: np.ndarray, interp: int) -> np.ndarray:
        mx = ((x + self.extent / 2) / self.res).astype(np.float32)
        my = ((y + self.extent / 2) / self.res).astype(np.float32)
        return cv2.remap(field_, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


def _noise_texture(rng: np.random.Generator, n: int, scales: tuple[int, ...]) -> np.ndarray:
    tex = np.zeros((n, n), np.float32)
    for s in scales:
        small = rng.random((n // s + 2, n // s + 2)).astype(np.float32)
        tex += cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)[:n, :n]
    tex -= tex.min()
    return tex / tex.max()


def _rot_xy(roll_deg: float, pitch_deg: float) -> np.ndarray:
    r, p = np.radians(roll_deg), np.radians(pitch_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return rx @ ry


def render_stereo(
    scene: Scene,
    rig: StereoRig,
    size: tuple[int, int] = (320, 240),
    altitude: float = 5.0,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    noise_sigma: float = 2.0,
    seed: int = 1,
    position: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Render a rectified (left, right) uint8 pair plus left-camera ground truth.

    position: the camera's (x, y) over the scene, in metres.
    Ground truth keys: depth (camera z, m), height (above ground, m),
    plane (nx, ny, nz, d) of the ground in the left camera frame.
    """
    r_wc, c_left = camera_pose(altitude, roll_deg, pitch_deg, position)
    c_right = c_left + r_wc @ np.array([rig.baseline, 0.0, 0.0])
    k = np.array([[rig.fx, 0, rig.cx], [0, rig.fy, rig.cy], [0, 0, 1]])

    rng = np.random.default_rng(seed)
    left, depth, hit_h = render_view(scene, k, None, size, r_wc, c_left, noise_sigma, rng)
    right, _, _ = render_view(scene, k, None, size, r_wc, c_right, noise_sigma, rng)
    n_c = r_wc.T @ np.array([0.0, 0.0, 1.0])
    return left, right, {"depth": depth, "height": hit_h, "plane": np.append(n_c, altitude)}


def camera_pose(altitude: float, roll_deg: float = 0.0, pitch_deg: float = 0.0,
                position: tuple[float, float] = (0.0, 0.0)):
    """(r_wc, centre) of a camera at (x, y, altitude) looking straight down, then tilted.

    Camera frame is OpenCV's (x right, y down, z forward); r_wc maps it to world.
    """
    r_wc = _rot_xy(roll_deg, pitch_deg) @ np.diag([1.0, -1.0, -1.0])
    return r_wc, np.array([position[0], position[1], altitude])


def render_view(
    scene: Scene,
    k: np.ndarray,
    dist: np.ndarray | None,
    size: tuple[int, int],
    r_wc: np.ndarray,
    centre: np.ndarray,
    noise_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one camera with intrinsics `k` and OpenCV distortion `dist` (None = pinhole).

    Returns (uint8 image, depth along the optical axis in m, height of the hit point in m).
    Distortion is exact: each pixel's ray comes from cv2.undistortPoints.
    """
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    if dist is None:
        x, y = (u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1]
    else:
        pix = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pix, k, np.asarray(dist, np.float64),
                                   criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9))
        x, y = norm[:, 0, 0].reshape(h, w), norm[:, 0, 1].reshape(h, w)
    rays_c = np.stack([x, y, np.ones_like(x)], -1)
    depth, hit_h, tex = _march(scene, rays_c, r_wc, centre)
    rng = rng or np.random.default_rng(0)
    img = 30 + 200 * tex + rng.normal(0, noise_sigma, tex.shape)
    return np.clip(img, 0, 255).astype(np.uint8), depth, hit_h


def _march(scene, rays_c, r_wc, cam_c, steps=400, refine=12):
    h, w = rays_c.shape[:2]
    rays_w = rays_c @ r_wc.T  # point at camera depth z is cam_c + z * rays_w

    hmax, hmin = float(scene.height.max()), float(scene.height.min())
    z_near = max(0.1, (cam_c[2] - hmax) * 0.7)
    z_far = (cam_c[2] - hmin) * 1.6

    def below(z):
        p = cam_c + z[..., None] * rays_w
        ground = scene.sample(p[..., 0], p[..., 1], scene.height, cv2.INTER_NEAREST)
        return p[..., 2] <= ground

    lo = np.full((h, w), z_near)
    hi = np.full((h, w), z_far)
    found = np.zeros((h, w), bool)
    prev = lo.copy()
    for z in np.linspace(z_near, z_far, steps):
        zz = np.full((h, w), z)
        hit = below(zz) & ~found
        lo[hit], hi[hit] = prev[hit], z
        found |= hit
        prev = zz
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        b = below(mid)
        hi = np.where(b, mid, hi)
        lo = np.where(b, lo, mid)

    depth = hi
    p = cam_c + depth[..., None] * rays_w
    tex = scene.sample(p[..., 0], p[..., 1], scene.texture, cv2.INTER_LINEAR)
    depth = np.where(found, depth, np.nan).astype(np.float32)
    return depth, p[..., 2].astype(np.float32), tex


def default_scene(seed: int = 0) -> Scene:
    """A ground plane with two structures and one pit, all in view from 5 m."""
    return Scene(
        seed=seed,
        blocks=[
            Block(x=1.0, y=0.4, w=1.6, l=1.2, h=1.5),
            Block(x=-1.4, y=-0.7, w=1.0, l=1.0, h=0.8),
            Block(x=0.3, y=-1.3, w=1.4, l=0.9, h=-0.8),
        ],
    )
