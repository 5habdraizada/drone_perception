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


def rectifying_rotations(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """(R1, R2, swapped): rotations taking each camera's coordinates into one shared rectified frame
    whose x axis runs along the baseline (Fusiello's construction), for any mounting, including
    cameras one behind the other or rotated relative to each other.

    x points from the left to the right camera of the rectified pair, choosing the direction
    closest to camera 1's own x axis; swapped is True when that makes camera 2 the left one.
    """
    centre2 = -np.asarray(r).T @ np.asarray(t).ravel()     # camera 2's centre in camera 1's frame
    x = centre2 / np.linalg.norm(centre2)
    swapped = bool(x[0] < 0)
    if swapped:
        x = -x
    z = np.array([0.0, 0.0, 1.0]) + np.asarray(r).T @ np.array([0.0, 0.0, 1.0])  # mean viewing direction
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    rect = np.stack([x, y, np.cross(x, y)])
    return rect, rect @ np.asarray(r).T, swapped


def refine_extrinsics(calib: StereoCalibration, pairs, mode: str = "rotation", max_features: int = 4000,
                      log=print) -> StereoCalibration:
    """Re-estimate the cameras' relative pose from feature matches in ordinary footage
    (pairs of raw (left, right) frames of a textured scene), keeping intrinsics and baseline length.

    mode "rotation" (default; for a calibrated rig that has drifted): keep the calibration's
        baseline direction and re-fit only the rotation. Well-posed even over flat ground.
    mode "full": estimate rotation and baseline direction from scratch (essential matrix, MAGSAC),
        for rigs with no trustworthy calibration. Over nearly flat scenes several geometries fit
        the matches about equally well, so the baseline direction can come out several degrees off.
    """
    if mode not in ("rotation", "full"):
        raise ValueError(f"unknown mode {mode!r}; use 'rotation' or 'full'")
    sift = cv2.SIFT_create(nfeatures=max_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pts_l, pts_r = [], []
    for l_img, r_img in pairs:
        kl, dl = sift.detectAndCompute(_gray(l_img), None)
        kr, dr = sift.detectAndCompute(_gray(r_img), None)
        if dl is None or dr is None:
            continue
        for m, n in (p for p in matcher.knnMatch(dl, dr, k=2) if len(p) == 2):
            if m.distance < 0.75 * n.distance:
                pts_l.append(kl[m.queryIdx].pt)
                pts_r.append(kr[m.trainIdx].pt)
    if len(pts_l) < 50:
        raise ValueError(f"only {len(pts_l)} feature matches; need a textured scene")
    nl = cv2.undistortPoints(np.float64(pts_l).reshape(-1, 1, 2), calib.left.K, calib.left.dist)
    nr = cv2.undistortPoints(np.float64(pts_r).reshape(-1, 1, 2), calib.right.K, calib.right.dist)
    px = 1.0 / np.mean([calib.left.K[0, 0], calib.right.K[0, 0]])  # 1 px in normalised units
    nl, nr = nl.reshape(-1, 2), nr.reshape(-1, 2)
    if mode == "full":
        e, mask = cv2.findEssentialMat(nl, nr, np.eye(3), method=cv2.USAC_MAGSAC, prob=0.9999, threshold=px)
        _, r, t, mask = cv2.recoverPose(e, nl, nr, np.eye(3), mask=mask)
        r, t, inl = r, t.ravel() * calib.baseline, mask.ravel() > 0
    else:
        r, inl = _refine_rotation(calib.R, calib.T / calib.baseline, nl, nr, px)
        t = calib.T.copy()
    out = StereoCalibration(calib.left, calib.right, r, t, math.nan, dict(calib.meta))

    before = _row_error(calib, np.float64(pts_l)[inl], np.float64(pts_r)[inl])
    after = _row_error(out, np.float64(pts_l)[inl], np.float64(pts_r)[inl])
    out.meta["refined_extrinsics"] = {"mode": mode, "matches": len(pts_l), "inliers": int(inl.sum()),
                                      "row_error_before_px": before, "row_error_after_px": after}
    log(f"refine_extrinsics ({mode}): {inl.sum()}/{len(pts_l)} inlier matches; rectified row error "
        f"{before:.2f} px -> {after:.2f} px (median, at {out.left.size[0]} px width)")
    return out


def _sampson(r: np.ndarray, t_dir: np.ndarray, nl: np.ndarray, nr: np.ndarray) -> np.ndarray:
    """Signed Sampson residual of each match under X_right = R X_left + t (normalised units)."""
    tx = np.array([[0, -t_dir[2], t_dir[1]], [t_dir[2], 0, -t_dir[0]], [-t_dir[1], t_dir[0], 0]])
    e = tx @ r
    a = np.column_stack([nl, np.ones(len(nl))])
    b = np.column_stack([nr, np.ones(len(nr))])
    ea, eb = a @ e.T, b @ e
    num = np.einsum("ij,ij->i", b, ea)
    return num / np.sqrt(ea[:, 0] ** 2 + ea[:, 1] ** 2 + eb[:, 0] ** 2 + eb[:, 1] ** 2 + 1e-18)


def _refine_rotation(r0: np.ndarray, t_dir: np.ndarray, nl: np.ndarray, nr: np.ndarray, px: float):
    """Levenberg-Marquardt on the rotation only (3 parameters), with the baseline direction fixed.
    Inliers are re-selected with a shrinking threshold: drift can start many pixels off."""
    r = np.asarray(r0, np.float64)
    inl = np.ones(len(nl), bool)
    for thr in (50 * px, 10 * px, 3 * px, 1.5 * px):
        inl = np.abs(_sampson(r, t_dir, nl, nr)) < thr
        if inl.sum() < 20:
            break
        lam = 1e-3
        for _ in range(20):
            res = _sampson(r, t_dir, nl[inl], nr[inl])
            jac = np.empty((len(res), 3))
            for k in range(3):
                d = np.zeros(3)
                d[k] = 1e-6
                jac[:, k] = (_sampson(cv2.Rodrigues(d)[0] @ r, t_dir, nl[inl], nr[inl]) - res) / 1e-6
            step = np.linalg.solve(jac.T @ jac + lam * np.eye(3), -jac.T @ res)
            cand = cv2.Rodrigues(step)[0] @ r
            if np.sum(_sampson(cand, t_dir, nl[inl], nr[inl]) ** 2) < np.sum(res ** 2):
                r, lam = cand, lam * 0.3
                if np.linalg.norm(step) < 1e-9:
                    break
            else:
                lam *= 10
    return r, np.abs(_sampson(r, t_dir, nl, nr)) < 1.5 * px


def _row_error(calib: StereoCalibration, pts_l: np.ndarray, pts_r: np.ndarray) -> float:
    """Median |row difference| of matched points after rectifying with `calib` (raw-resolution pixels)."""
    r1, r2, _ = rectifying_rotations(calib.R, calib.T)
    k = calib.left.K
    pl = cv2.undistortPoints(pts_l.reshape(-1, 1, 2), calib.left.K, calib.left.dist, R=r1, P=k).reshape(-1, 2)
    pr = cv2.undistortPoints(pts_r.reshape(-1, 1, 2), calib.right.K, calib.right.dist, R=r2, P=k).reshape(-1, 2)
    return float(np.median(np.abs(pl[:, 1] - pr[:, 1])))


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _field_polygon(model: CameraModel, r: np.ndarray) -> np.ndarray:
    """A camera's usable image border (clipped to the distortion model's valid radius),
    as a polygon in rectified normalised coordinates."""
    w, h = model.size
    t = np.linspace(0, 1, 64)[:-1]
    border = np.concatenate([np.stack([t * (w - 1), np.zeros_like(t)], -1),
                             np.stack([np.full_like(t, w - 1), t * (h - 1)], -1),
                             np.stack([(1 - t) * (w - 1), np.full_like(t, h - 1)], -1),
                             np.stack([np.zeros_like(t), (1 - t) * (h - 1)], -1)])
    limit = valid_radius(model)
    if math.isfinite(limit):
        f, c = np.array([model.K[0, 0], model.K[1, 1]]), model.K[:2, 2]
        norm = (border - c) / f
        rad = np.linalg.norm(norm, axis=1, keepdims=True)
        border = np.where(rad > limit, norm * limit / np.maximum(rad, 1e-12), norm) * f + c
    return cv2.undistortPoints(border.reshape(-1, 1, 2), model.K, model.dist, R=r).reshape(-1, 2)


def _largest_rectangle(inside: np.ndarray, aspects) -> tuple[int, int, int, int] | None:
    """(r0, c0, r1, c1) of the largest rectangle of one of the given width/height aspects
    lying entirely inside the mask; candidate centres are the mask's most interior points."""
    integral = cv2.integral((1 - inside).astype(np.uint8))  # counts of outside pixels
    outside = lambda r0, c0, r1, c1: integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    order = np.argsort(dist.ravel())[::-1][:400:20]
    centres = [np.unravel_index(i, dist.shape) for i in order]
    ys, xs = np.nonzero(inside)
    centres.append((int(ys.mean()), int(xs.mean())))
    hgt, wid = inside.shape
    best, best_area = None, 0.0
    for a in aspects:
        for cy, cx in centres:
            lo_h, hi_h = 0.0, float(max(hgt, wid))
            for _ in range(20):  # binary search on the half-height
                mid = (lo_h + hi_h) / 2
                r0, r1 = int(cy - mid), int(np.ceil(cy + mid))
                c0, c1 = int(cx - a * mid), int(np.ceil(cx + a * mid))
                ok = r0 >= 0 and c0 >= 0 and r1 < hgt and c1 < wid and outside(r0, c0, r1 + 1, c1 + 1) == 0
                lo_h, hi_h = (mid, hi_h) if ok else (lo_h, mid)
            area = 4 * a * lo_h * lo_h
            if lo_h > 0 and area > best_area:
                best_area = area
                best = (int(cy - lo_h), int(cx - a * lo_h), int(np.ceil(cy + lo_h)), int(np.ceil(cx + a * lo_h)))
    return best


class Rectifier:
    """Warps raw (left, right) frames into a rectified pair of `out_size`.

    Every output pixel lies inside both cameras' views. With out_size = (w, None)
    the height follows the shape of the overlap (rounded to 16 px), so nothing
    is cropped or wasted; with an explicit (w, h) of a different shape, the
    overlap is cropped to fill it.

    Any mounting works: the rectified x axis runs along the baseline. The pair is returned
    left-to-right along it; if that makes camera 2 the left one, `swapped` is True.
    """

    def __init__(self, calib: StereoCalibration, out_size: tuple[int, int | None] = (640, None)):
        self.calib = calib
        cl, cr = calib.left, calib.right
        for name, model in (("left", cl), ("right", cr)):
            if not model_is_usable(model):
                raise ValueError(f"{name} camera's distortion model is not valid across the image; "
                                 f"recalibrate it (calibrate_stereo.py refines intrinsics automatically): "
                                 + check_model(model)[0])

        r1, r2, self.swapped = rectifying_rotations(calib.R, calib.T)
        self.baseline = calib.baseline

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
        """Largest axis-aligned rectangle (in rectified normalised coordinates) inside both cameras'
        valid fields, whatever their orientation, then the pinhole that maps it onto out_size."""
        polys = [_field_polygon(model, r) for model, r in cams]
        pts = np.concatenate(polys)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        n = 600
        scale = (n - 1) / float((hi - lo).max())
        shape = (int((hi[1] - lo[1]) * scale) + 2, int((hi[0] - lo[0]) * scale) + 2)
        inside = np.ones(shape, np.uint8)
        for poly in polys:
            m = np.zeros(shape, np.uint8)
            cv2.fillPoly(m, [np.round((poly - lo) * scale).astype(np.int32)], 1)
            inside &= m
        inside = cv2.erode(inside, np.ones((3, 3), np.uint8))  # stay clear of the rasterised border
        if inside.sum() < 100:
            raise ValueError("the two cameras' views do not overlap")
        w, h = out_size
        aspects = [w / h] if h is not None else np.geomspace(0.4, 4.0, 25)
        best = _largest_rectangle(inside, aspects)
        if best is None:
            raise ValueError("the two cameras' views do not overlap")
        r0, c0, r1, c1 = best
        x0, x1 = lo[0] + c0 / scale, lo[0] + c1 / scale
        y0, y1 = lo[1] + r0 / scale, lo[1] + r1 / scale
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
