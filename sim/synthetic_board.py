"""Synthetic checkerboard captures through real (distorted) camera models.

Used to test the calibration tool end to end on images: detection, intrinsic
refinement, stereo calibration and rectification.
"""
from __future__ import annotations

import cv2
import numpy as np

from perception.calibration import BoardSpec, CameraModel


class BoardRenderer:
    """Renders a printed checkerboard (with a one-square white margin) into one camera."""

    def __init__(self, model: CameraModel, board: BoardSpec, px_per_square: int = 40):
        self.model, self.board, self.pps = model, board, px_per_square
        squares = np.indices((board.rows + 1, board.cols + 1)).sum(axis=0) % 2
        # Colouring chosen so OpenCV's detection order matches BoardSpec.object_points().
        tex = np.kron(1 - squares, np.ones((px_per_square, px_per_square))) * 215 + 25
        self.texture = cv2.copyMakeBorder(tex.astype(np.float32), px_per_square, px_per_square,
                                          px_per_square, px_per_square, cv2.BORDER_CONSTANT, value=240)
        w, h = model.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pix = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pix, model.K, model.dist,
                                   criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9))
        self.rays = np.concatenate([norm.reshape(h, w, 2), np.ones((h, w, 1))], axis=-1)

    def render(self, rvec: np.ndarray, tvec: np.ndarray, rng: np.random.Generator,
               noise: float = 2.0) -> np.ndarray:
        rot = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
        t = np.asarray(tvec, np.float64).ravel()
        n = rot[:, 2]
        denom = self.rays @ n
        s = (n @ t) / np.where(np.abs(denom) < 1e-12, np.nan, denom)
        hit = (s[..., None] * self.rays - t) @ rot  # back into board coordinates
        # Inner corner (0, 0) sits two squares in from the textured image's corner;
        # -0.5 because a square's edge lies between texture pixels, whose centres are integers.
        mx = ((hit[..., 0] / self.board.square + 2) * self.pps - 0.5).astype(np.float32)
        my = ((hit[..., 1] / self.board.square + 2) * self.pps - 0.5).astype(np.float32)
        bad = ~np.isfinite(s) | (s <= 0)
        mx[bad], my[bad] = -1, -1
        img = cv2.remap(self.texture, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=110)
        img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def sample_poses(model: CameraModel, board: BoardSpec, n: int, rng: np.random.Generator,
                 depth=(0.9, 1.8), spread=1.0, visible_in=None, invisible_in=None):
    """Random board poses in `model`'s frame, fully in view of `model`.

    visible_in / invisible_in: extra (CameraModel, R, T) cameras, with
    X_cam = R @ X_model + T, that must also fully see / must not fully see the board.
    `spread` scales how far towards the image edges the board is placed.
    """
    outline = board_outline(board)
    centre = outline.mean(axis=0)
    poses = []
    while len(poses) < n:
        rvec = np.radians(rng.uniform([-35, -35, -25], [35, 35, 25]))
        z = rng.uniform(*depth)
        fx_reach = model.size[0] / 2 / model.K[0, 0] * spread
        fy_reach = model.size[1] / 2 / model.K[1, 1] * spread
        tvec = np.array([rng.uniform(-fx_reach, fx_reach) * z, rng.uniform(-fy_reach, fy_reach) * z, z])
        tvec = tvec - cv2.Rodrigues(rvec)[0] @ centre
        if not _sees(model, outline, rvec, tvec):
            continue
        rot = cv2.Rodrigues(rvec)[0]
        ok = all(_sees(m, outline, cv2.Rodrigues(r @ rot)[0], r @ tvec + t) for m, r, t in (visible_in or []))
        ok &= not any(_sees(m, outline, cv2.Rodrigues(r @ rot)[0], r @ tvec + t) for m, r, t in (invisible_in or []))
        if ok:
            poses.append((rvec, tvec))
    return poses


def board_outline(board: BoardSpec) -> np.ndarray:
    """Board corners including the white margin, in board coordinates (m)."""
    s = board.square
    x0, y0 = -2 * s, -2 * s
    x1, y1 = (board.cols + 1) * s, (board.rows + 1) * s
    return np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], np.float64)


def _sees(model: CameraModel, outline: np.ndarray, rvec, tvec, margin: int = 15) -> bool:
    rot = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    if ((outline @ rot.T + tvec)[:, 2] <= 0.1).any():
        return False
    p = cv2.projectPoints(outline, rvec, tvec, model.K, model.dist)[0].reshape(-1, 2)
    w, h = model.size
    return bool(((p[:, 0] > margin) & (p[:, 0] < w - margin) & (p[:, 1] > margin) & (p[:, 1] < h - margin)).all())
