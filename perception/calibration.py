"""Stereo calibration and rectification, including mismatched lenses.

find_board -> stereo_calibrate (X_right = R @ X_left + T, metres) -> Rectifier.
check_model flags overfit distortion models that can't undistort the whole image.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from perception.terrain import StereoRig

_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-9)


@dataclass
class CameraModel:
    """Pinhole intrinsics plus OpenCV distortion, valid at image `size` (w, h)."""
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]

    @classmethod
    def from_values(cls, fx: float, fy: float, cx: float, cy: float,
                    size: tuple[int, int], dist=None) -> "CameraModel":
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return cls(k, np.zeros(5) if dist is None else np.asarray(dist, np.float64), tuple(size))

    def scaled(self, size: tuple[int, int]) -> "CameraModel":
        """Same camera at another resolution (pixel-centre convention preserved)."""
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        k = self.K.copy()
        k[0, 0] *= sx
        k[1, 1] *= sy
        k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
        k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
        return CameraModel(k, self.dist.copy(), tuple(size))

    @property
    def fov_deg(self) -> tuple[float, float]:
        w, h = self.size
        return (math.degrees(2 * math.atan(w / 2 / self.K[0, 0])),
                math.degrees(2 * math.atan(h / 2 / self.K[1, 1])))

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(), "size": list(self.size)}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(np.array(d["K"], np.float64), np.array(d["dist"], np.float64), tuple(d["size"]))


def check_model(model: CameraModel) -> list[str]:
    """Problems with a camera model that would make it unsafe to use; empty if fine.

    The usual failure is an overfit distortion model: when the calibration board
    never reached the image edges, high-order terms fit noise and the model
    folds back on itself before the corners, so edge pixels can't be undistorted.
    """
    issues = []
    p1, p2 = model.dist.ravel()[2:4]
    w, h = model.size
    cx, cy = model.K[0, 2], model.K[1, 2]

    bad = invalid_fraction(model)
    if bad > 0:
        severity = "unusable" if bad > MAX_INVALID_FRACTION else "only the outer rim is affected"
        issues.append(f"distortion model folds back before the image edge: {bad:.1%} of the image "
                      f"can't be undistorted ({severity}); the model is overfit there, usually "
                      f"because the board never reached those parts of the image")
    if max(abs(p1), abs(p2)) > 0.005:
        issues.append(f"tangential distortion is large (p1={p1:+.4f}, p2={p2:+.4f}); "
                      f"usually a sign of an under-constrained calibration")
    off = max(abs(cx - (w - 1) / 2) / w, abs(cy - (h - 1) / 2) / h)
    if off > 0.06:
        issues.append(f"principal point ({cx:.0f}, {cy:.0f}) is {off:.0%} of the image from the centre; "
                      f"possible, but worth confirming with a well-spread calibration")
    return issues


MAX_INVALID_FRACTION = 0.10  # more of the image than this beyond the model's valid radius = unusable


def valid_radius(model: CameraModel) -> float:
    """Largest distorted, normalised radius the radial model can produce (inf if it never folds).

    Beyond it the polynomial turns back on itself, so those pixels have no
    undistorted position. Includes a 2% safety margin.
    """
    d = model.dist.ravel()
    if d.size > 5:  # rational model: has a denominator, the simple fold test doesn't apply
        return math.inf
    k1, k2 = d[0], d[1]
    k3 = d[4] if d.size > 4 else 0.0
    r = np.linspace(0.0, 2.0, 4001)
    rd = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    falling = np.diff(rd) <= 0
    return float(rd[np.argmax(falling)] * 0.98) if falling.any() else math.inf


def invalid_fraction(model: CameraModel) -> float:
    """Fraction of the image beyond valid_radius()."""
    limit = valid_radius(model)
    if not math.isfinite(limit):
        return 0.0
    w, h = model.size
    u, v = np.meshgrid(np.linspace(0, w - 1, 160), np.linspace(0, h - 1, 90))
    rad = np.hypot((u - model.K[0, 2]) / model.K[0, 0], (v - model.K[1, 2]) / model.K[1, 1])
    return float(np.mean(rad > limit))


def model_is_usable(model: CameraModel) -> bool:
    """False if too much of the image lies where the distortion model is invalid."""
    return invalid_fraction(model) <= MAX_INVALID_FRACTION


@dataclass
class StereoCalibration:
    left: CameraModel
    right: CameraModel
    R: np.ndarray                  # 3×3, left camera frame -> right camera frame
    T: np.ndarray                  # 3, metres: X_right = R @ X_left + T
    rms: float = math.nan          # reprojection RMS over both cameras, px
    meta: dict = field(default_factory=dict)

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def rotation_deg(self) -> float:
        """Angle of the relative rotation between the two cameras."""
        return float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R)[0])))

    def save(self, path: str | Path) -> None:
        data = {"left": self.left.to_dict(), "right": self.right.to_dict(),
                "R": self.R.tolist(), "T": self.T.ravel().tolist(), "rms": self.rms, "meta": self.meta}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "StereoCalibration":
        d = json.loads(Path(path).read_text())
        return cls(CameraModel.from_dict(d["left"]), CameraModel.from_dict(d["right"]),
                   np.array(d["R"], np.float64), np.array(d["T"], np.float64),
                   float(d.get("rms", math.nan)), d.get("meta", {}))


@dataclass(frozen=True)
class BoardSpec:
    """Checkerboard counted in INNER corners. Use one odd and one even count
    (e.g. 9 × 6) so the detected corner order can't flip by 180 degrees."""
    cols: int = 9
    rows: int = 6
    square: float = 0.025  # metres, measured on the printed board

    def object_points(self) -> np.ndarray:
        grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        return np.hstack([grid * self.square, np.zeros((len(grid), 1))]).astype(np.float32)


def find_board(img: np.ndarray, board: BoardSpec) -> np.ndarray | None:
    """Sub-pixel checkerboard corners as an N×1×2 float32 array, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    pattern = (board.cols, board.rows)
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), _CRITERIA)
    # The two detectors (and OpenCV versions) disagree on N×2 vs N×1×2.
    return corners.reshape(-1, 1, 2).astype(np.float32)


def calibrate_camera(board: BoardSpec, corners: list[np.ndarray], guess: CameraModel,
                     rational: bool = False) -> tuple[CameraModel, float]:
    """Single-camera calibration starting from `guess`. Returns (model, rms px)."""
    obj = [board.object_points()] * len(corners)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_RATIONAL_MODEL if rational else 0)
    dist = np.zeros(8 if rational else 5)
    dist[: min(len(dist), guess.dist.size)] = guess.dist.ravel()[: len(dist)]
    rms, k, d, _, _ = cv2.calibrateCamera(obj, corners, guess.size, guess.K.copy(), dist,
                                          flags=flags, criteria=_CRITERIA)
    return CameraModel(k, d.ravel(), guess.size), float(rms)


def stereo_calibrate(board: BoardSpec, left_corners: list[np.ndarray], right_corners: list[np.ndarray],
                     left: CameraModel, right: CameraModel,
                     refine_intrinsics: bool = False, rational: bool = False) -> StereoCalibration:
    """Relative pose of the right camera. With refine_intrinsics, each camera is
    first re-calibrated from the same captures (starting from the given values),
    then the pair is solved with those intrinsics held fixed, which is more
    stable than solving everything jointly."""
    if len(left_corners) != len(right_corners) or len(left_corners) < 3:
        raise ValueError("need at least 3 synchronised pairs with the board found in both")
    if refine_intrinsics:
        left, _ = calibrate_camera(board, left_corners, left, rational)
        right, _ = calibrate_camera(board, right_corners, right, rational)
    obj = [board.object_points()] * len(left_corners)
    rms, _, _, _, _, r, t, _, _ = cv2.stereoCalibrate(
        obj, left_corners, right_corners, left.K, left.dist, right.K, right.dist, left.size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=_CRITERIA)
    return StereoCalibration(left, right, r, t.ravel(), float(rms))


def per_pair_errors(calib: StereoCalibration, board: BoardSpec,
                    left_corners: list[np.ndarray], right_corners: list[np.ndarray]) -> np.ndarray:
    """RMS reprojection error of each pair (px), to spot bad captures."""
    obj = board.object_points()
    errs = []
    for cl, cr in zip(left_corners, right_corners):
        _, rvec, tvec = cv2.solvePnP(obj, cl, calib.left.K, calib.left.dist)
        rl = cv2.Rodrigues(rvec)[0]
        rr, tr = calib.R @ rl, calib.R @ tvec.ravel() + calib.T
        pl = cv2.projectPoints(obj, rvec, tvec, calib.left.K, calib.left.dist)[0]
        pr = cv2.projectPoints(obj, cv2.Rodrigues(rr)[0], tr, calib.right.K, calib.right.dist)[0]
        res = np.concatenate([(pl - cl).reshape(-1, 2), (pr - cr).reshape(-1, 2)])
        errs.append(float(np.sqrt((res ** 2).sum(axis=1).mean())))
    return np.array(errs)


class Rectifier:
    """Warps raw (left, right) frames into a rectified pair of `out_size`.

    Every output pixel lies inside both cameras' views. With out_size = (w, None)
    the height follows the shape of the overlap (rounded to 16 px), so nothing
    is cropped or wasted; with an explicit (w, h) of a different shape, the
    overlap is cropped to fill it.

    The rectified pair is always returned in left-to-right order along the
    baseline. If camera 2 turns out to sit on the left of camera 1, the
    outputs are swapped and `swapped` is True; the rectified reference frame is
    then camera 2's.
    """

    def __init__(self, calib: StereoCalibration, out_size: tuple[int, int | None] = (640, None)):
        self.calib = calib
        cl, cr = calib.left, calib.right
        for name, model in (("left", cl), ("right", cr)):
            if not model_is_usable(model):
                raise ValueError(f"{name} camera's distortion model is not valid across the image; "
                                 f"recalibrate it (calibrate_stereo.py refines intrinsics automatically): "
                                 + check_model(model)[0])

        # Rotations only depend on R and T, so the shared image size argument is irrelevant here.
        r1, r2, *_ = cv2.stereoRectify(cl.K, cl.dist, cr.K, cr.dist, cl.size, calib.R,
                                       calib.T.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        # Right camera centre, expressed in the rectified frame of camera 1.
        centre = r1 @ (-calib.R.T @ calib.T)
        if abs(centre[1]) > abs(centre[0]):
            raise ValueError("cameras are stacked vertically in the image; terrain.py needs them "
                             "side by side along image rows (rotate both images 90 degrees)")
        self.swapped = bool(centre[0] < 0)
        self.baseline = float(abs(centre[0]))

        # One shared pinhole for both, covering only the view valid in both cameras.
        f, cx, cy, self.out_size = self._common_pinhole([(cl, r1), (cr, r2)], out_size)
        k_new = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.K = k_new
        self.R1, self.R2 = r1, r2

        self._views = [self._view(cl, r1, k_new), self._view(cr, r2, k_new)]
        self.rig = StereoRig(fx=f, fy=f, cx=cx, cy=cy, baseline=self.baseline)

    def rectify(self, left: np.ndarray, right: np.ndarray, gray: bool = True) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for img, (model, pre_size, m1, m2) in zip((left, right), self._views):
            if (img.shape[1], img.shape[0]) != model.size:
                raise ValueError(f"image is {img.shape[1]}x{img.shape[0]}, calibration is "
                                 f"{model.size[0]}x{model.size[1]}")
            if gray and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if pre_size != model.size:
                img = cv2.resize(img, pre_size, interpolation=cv2.INTER_AREA)
            out.append(cv2.remap(img, m1, m2, cv2.INTER_LINEAR))
        return (out[1], out[0]) if self.swapped else (out[0], out[1])

    def rectify_points(self, left_pts: np.ndarray, right_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw pixel points -> rectified pixels, in the same order rectify() returns images."""
        cl, cr = self.calib.left, self.calib.right
        pl = cv2.undistortPoints(left_pts.reshape(-1, 1, 2), cl.K, cl.dist, R=self.R1, P=self.K)
        pr = cv2.undistortPoints(right_pts.reshape(-1, 1, 2), cr.K, cr.dist, R=self.R2, P=self.K)
        pl, pr = pl.reshape(-1, 2), pr.reshape(-1, 2)
        return (pr, pl) if self.swapped else (pl, pr)

    @staticmethod
    def _common_pinhole(cams, out_size) -> tuple[float, float, float, tuple[int, int]]:
        # Inner bounds of each camera's image border after undistortion and rotation,
        # in normalised rectified coordinates; intersect them over both cameras.
        x0, y0, x1, y1 = -np.inf, -np.inf, np.inf, np.inf
        for model, r in cams:
            w, h = model.size
            t = np.linspace(0, 1, 64)
            edges = {
                "left": np.stack([np.zeros_like(t), t * (h - 1)], -1),
                "right": np.stack([np.full_like(t, w - 1), t * (h - 1)], -1),
                "top": np.stack([t * (w - 1), np.zeros_like(t)], -1),
                "bottom": np.stack([t * (w - 1), np.full_like(t, h - 1)], -1),
            }
            # Pull border points that lie beyond the model's valid radius in to it, so
            # the usable field is the image intersected with the valid disc.
            limit = valid_radius(model)
            if math.isfinite(limit):
                f, c = np.array([model.K[0, 0], model.K[1, 1]]), model.K[:2, 2]
                for k, pts in edges.items():
                    norm = (pts - c) / f
                    rad = np.linalg.norm(norm, axis=1, keepdims=True)
                    edges[k] = np.where(rad > limit, norm * limit / np.maximum(rad, 1e-12), norm) * f + c
            n = {k: cv2.undistortPoints(v.reshape(-1, 1, 2), model.K, model.dist, R=r).reshape(-1, 2)
                 for k, v in edges.items()}
            x0, x1 = max(x0, n["left"][:, 0].max()), min(x1, n["right"][:, 0].min())
            y0, y1 = max(y0, n["top"][:, 1].max()), min(y1, n["bottom"][:, 1].min())
        if not (x1 > x0 and y1 > y0):
            raise ValueError("the two cameras' views do not overlap")
        w, h = out_size
        if h is None:
            h = max(16, int(round(w * (y1 - y0) / (x1 - x0) / 16)) * 16)
        # The larger scale fills the output, cropping the overlap along the other axis.
        f = max(w / (x1 - x0), h / (y1 - y0))
        return f, w / 2 - f * (x0 + x1) / 2, h / 2 - f * (y0 + y1) / 2, (w, h)

    def _view(self, model: CameraModel, r: np.ndarray, k_new: np.ndarray):
        # Shrink the raw image to about the rectified sampling first (never enlarge).
        shrink = min(1.0, k_new[0, 0] / model.K[0, 0])
        pre_size = (max(1, round(model.size[0] * shrink)), max(1, round(model.size[1] * shrink)))
        scaled = model.scaled(pre_size)
        m1, m2 = cv2.initUndistortRectifyMap(scaled.K, scaled.dist, r, k_new, self.out_size, cv2.CV_16SC2)
        return model, pre_size, m1, m2
