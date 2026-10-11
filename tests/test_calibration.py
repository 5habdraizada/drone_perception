import math

import cv2
import numpy as np
import pytest

from perception.calibration import (BoardSpec, CameraModel, Rectifier, StereoCalibration,
                                    find_board, per_pair_errors, refine_extrinsics, stereo_calibrate)
from perception.terrain import TerrainConfig, TerrainEstimator
from sim.synthetic_stereo import camera_pose, default_scene, render_view

SIZE = (1920, 1080)
# The project's real pair: a long (narrow) left lens and a short (wide) right lens.
# Distortion and extrinsics are made up but plausible.
LEFT = CameraModel.from_values(1718.54, 1720.12, 1054.17, 454.72, SIZE, [-0.03, 0.02, 0.0005, -0.0003, 0.0])
RIGHT = CameraModel.from_values(1175.14, 1177.06, 919.18, 602.39, SIZE, [-0.12, 0.05, -0.0008, 0.0006, -0.01])
BOARD = BoardSpec(9, 6, 0.04)


def _rot(rx, ry, rz):
    return cv2.Rodrigues(np.radians([rx, ry, rz]))[0]


def _true_calib(right_centre=(0.10, 0.002, -0.001)):
    r = _rot(0.8, -0.5, 0.3)                       # small mounting misalignment
    t = -r @ np.asarray(right_centre, np.float64)   # X_right = R (X_left - c)
    return StereoCalibration(LEFT, RIGHT, r, t)


def _observations(calib, n=25, noise=0.2, seed=0):
    """Checkerboard corners seen by both cameras from random poses, fully in view."""
    rng = np.random.default_rng(seed)
    obj = BOARD.object_points()
    centre = obj.mean(axis=0)
    lefts, rights = [], []
    while len(lefts) < n:
        rvec = np.radians(rng.uniform([-30, -30, -20], [30, 30, 20]))
        tvec = np.array([rng.uniform(-0.25, 0.3), rng.uniform(-0.25, 0.15), rng.uniform(0.9, 1.8)])
        tvec = tvec - cv2.Rodrigues(rvec)[0] @ centre
        pl = cv2.projectPoints(obj, rvec, tvec, calib.left.K, calib.left.dist)[0]
        rr = calib.R @ cv2.Rodrigues(rvec)[0]
        pr = cv2.projectPoints(obj, cv2.Rodrigues(rr)[0], calib.R @ tvec + calib.T,
                               calib.right.K, calib.right.dist)[0]
        inside = all(((p[..., 0] > 20) & (p[..., 0] < SIZE[0] - 20) &
                      (p[..., 1] > 20) & (p[..., 1] < SIZE[1] - 20)).all() for p in (pl, pr))
        if inside:
            lefts.append((pl + rng.normal(0, noise, pl.shape)).astype(np.float32))
            rights.append((pr + rng.normal(0, noise, pr.shape)).astype(np.float32))
    return lefts, rights


def _rot_err_deg(a, b):
    return math.degrees(np.linalg.norm(cv2.Rodrigues(a @ b.T)[0]))


def test_recovers_extrinsics_with_known_intrinsics():
    truth = _true_calib()
    lefts, rights = _observations(truth)
    calib = stereo_calibrate(BOARD, lefts, rights, LEFT, RIGHT)
    assert abs(calib.baseline - truth.baseline) < 0.001
    assert _rot_err_deg(calib.R, truth.R) < 0.05
    assert calib.rms < 0.4
    assert per_pair_errors(calib, BOARD, lefts, rights).max() < 0.5


def test_refines_rough_intrinsics_and_unknown_distortion():
    truth = _true_calib()
    lefts, rights = _observations(truth, n=30)
    # Start from slightly wrong focal lengths and no distortion at all.
    rough_l = CameraModel.from_values(1718.54 * 1.02, 1720.12 * 1.02, 1054.17, 454.72, SIZE)
    rough_r = CameraModel.from_values(1175.14 * 0.98, 1177.06 * 0.98, 919.18, 602.39, SIZE)
    calib = stereo_calibrate(BOARD, lefts, rights, rough_l, rough_r, refine_intrinsics=True)
    assert calib.rms < 0.4
    assert abs(calib.baseline - truth.baseline) < 0.002
    assert abs(calib.right.K[0, 0] - RIGHT.K[0, 0]) / RIGHT.K[0, 0] < 0.005


def test_rectification_aligns_rows_and_gives_positive_disparity():
    truth = _true_calib()
    lefts, rights = _observations(truth, noise=0.0)
    rect = Rectifier(truth, (640, 480))
    assert not rect.swapped
    assert rect.baseline == pytest.approx(0.10, abs=1e-3)
    pl, pr = rect.rectify_points(np.concatenate(lefts), np.concatenate(rights))
    assert np.abs(pl[:, 1] - pr[:, 1]).max() < 0.05
    assert (pl[:, 0] - pr[:, 0]).min() > 0


def test_default_output_follows_the_overlap():
    rect = Rectifier(_true_calib())
    # Overlap is bounded by the long lens (~58 x 35 deg, 16:9), not the wide one,
    # so the default 640-wide output comes out widescreen rather than 4:3.
    w, h = rect.out_size
    assert w == 640 and h % 16 == 0 and 330 <= h <= 380
    hfov = math.degrees(2 * math.atan(320 / rect.K[0, 0]))
    assert 50 < hfov < LEFT.fov_deg[0] + 1


def test_every_output_pixel_is_inside_both_views():
    rect = Rectifier(_true_calib(), (640, 480))  # 4:3 forces a crop of the 16:9 overlap
    left = np.full(SIZE[::-1], 200, np.uint8)
    rl, rr = rect.rectify(left, left.copy())
    # Remap fills anything outside a source image with 0; allow a 1 px interpolation rim.
    assert (rl[1:-1, 1:-1] > 0).all() and (rr[1:-1, 1:-1] > 0).all()


def test_cameras_mounted_the_other_way_round_are_swapped():
    swapped = _true_calib(right_centre=(-0.10, 0.002, -0.001))
    lefts, rights = _observations(swapped, noise=0.0)
    rect = Rectifier(swapped, (640, 480))
    assert rect.swapped
    pl, pr = rect.rectify_points(np.concatenate(lefts), np.concatenate(rights))
    assert (pl[:, 0] - pr[:, 0]).min() > 0


@pytest.mark.parametrize("centre, roll_deg", [((0.0, 0.10, 0.0), 0.0),    # one above the other
                                              ((0.0, 0.10, 0.0), 90.0),   # and rotated 90 deg, like Kagaru
                                              ((0.07, 0.07, 0.0), 30.0)])
def test_any_mounting_rectifies(centre, roll_deg):
    calib = _true_calib(right_centre=centre)
    r = cv2.Rodrigues(np.radians([0.0, 0.0, roll_deg]))[0] @ calib.R
    calib = StereoCalibration(LEFT, RIGHT, r, -r @ np.asarray(centre))
    lefts, rights = _observations(calib, noise=0.0)
    rect = Rectifier(calib, (640, None))
    pl, pr = rect.rectify_points(np.concatenate(lefts), np.concatenate(rights))
    assert np.abs(pl[:, 1] - pr[:, 1]).max() < 0.05
    assert (pl[:, 0] - pr[:, 0]).min() > 0
    assert rect.baseline == pytest.approx(np.linalg.norm(centre), abs=1e-6)


@pytest.mark.slow
def test_refine_extrinsics_fixes_a_drifted_calibration():
    # Footage rendered with the true calibration; the "known" calibration has drifted by ~1 deg.
    half = (960, 540)
    truth = _true_calib()
    truth = StereoCalibration(truth.left.scaled(half), truth.right.scaled(half), truth.R, truth.T)
    scene = default_scene()
    rng = np.random.default_rng(0)
    pairs = []
    for x in (-0.5, 0.5):
        r_wc, c = camera_pose(5.0, 2.0, -1.0, (x, 0.0))
        left, _, _ = render_view(scene, truth.left.K, truth.left.dist, half, r_wc, c, rng=rng)
        right, _, _ = render_view(scene, truth.right.K, truth.right.dist, half, r_wc @ truth.R.T,
                                  c + r_wc @ (-truth.R.T @ truth.T), rng=rng)
        pairs.append((left, right))
    drift = cv2.Rodrigues(np.radians([0.6, -0.5, 0.4]))[0]
    drifted = StereoCalibration(truth.left, truth.right, drift @ truth.R, drift @ truth.T)
    fixed = refine_extrinsics(drifted, pairs, log=lambda *_: None)
    err = lambda c: math.degrees(np.linalg.norm(cv2.Rodrigues(c.R @ truth.R.T)[0]))
    # What stereo needs is row alignment; rotation about the baseline barely affects it.
    assert err(drifted) > 0.8 and err(fixed) < 0.25
    assert fixed.baseline == pytest.approx(truth.baseline)
    info = fixed.meta["refined_extrinsics"]
    assert info["row_error_before_px"] > 3 and info["row_error_after_px"] < 0.3


def test_save_load_roundtrip(tmp_path):
    calib = _true_calib()
    calib.rms = 0.21
    calib.meta = {"pairs": 25}
    calib.save(tmp_path / "stereo.json")
    back = StereoCalibration.load(tmp_path / "stereo.json")
    assert np.allclose(back.R, calib.R) and np.allclose(back.T, calib.T)
    assert np.allclose(back.left.K, LEFT.K) and np.allclose(back.right.dist, RIGHT.dist)
    assert back.left.size == SIZE and back.rms == 0.21 and back.meta == {"pairs": 25}


def test_find_board_on_rendered_checkerboard():
    sq = 40
    cells = np.indices((BOARD.rows + 1, BOARD.cols + 1)).sum(axis=0) % 2
    board_img = np.kron(cells, np.ones((sq, sq))).astype(np.uint8) * 255
    canvas = np.full((600, 800), 180, np.uint8)
    h, w = board_img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[150, 120], [620, 90], [650, 470], [130, 450]])
    warp = cv2.getPerspectiveTransform(src, dst)
    mask = cv2.warpPerspective(np.ones_like(board_img), warp, (800, 600)) > 0
    canvas[mask] = cv2.warpPerspective(board_img, warp, (800, 600))[mask]
    corners = find_board(canvas, BOARD)
    assert corners is not None and len(corners) == BOARD.cols * BOARD.rows
    # The first inner corner is one square in from the board's top-left.
    expected = cv2.perspectiveTransform(np.float32([[[sq, sq]]]), warp)[0, 0]
    assert np.linalg.norm(corners.reshape(-1, 2) - expected, axis=1).min() < 1.0


def test_find_board_returns_none_without_board():
    assert find_board(np.full((480, 640), 128, np.uint8), BOARD) is None


@pytest.fixture(scope="module")
def raw_mismatched_pair():
    # Render the real (mismatched, distorted) cameras at half resolution to keep the test quick.
    half = (960, 540)
    calib = _true_calib()
    calib = StereoCalibration(calib.left.scaled(half), calib.right.scaled(half), calib.R, calib.T)
    r_wc, c_left = camera_pose(5.0, roll_deg=3, pitch_deg=-2)
    r_wc_right = r_wc @ calib.R.T
    c_right = c_left + r_wc @ (-calib.R.T @ calib.T)
    rng = np.random.default_rng(3)
    scene = default_scene()
    left, _, _ = render_view(scene, calib.left.K, calib.left.dist, half, r_wc, c_left, rng=rng)
    right, _, _ = render_view(scene, calib.right.K, calib.right.dist, half, r_wc_right, c_right, rng=rng)
    return calib, scene, r_wc, c_left, left, right


@pytest.mark.slow
def test_end_to_end_mismatched_pair_into_terrain(raw_mismatched_pair):
    calib, scene, r_wc, c_left, left, right = raw_mismatched_pair
    rect = Rectifier(calib, (640, 480))
    rl, rr = rect.rectify(left, right)
    terrain = TerrainEstimator(rect.rig, TerrainConfig.for_width(640)).process(rl, rr)

    # Ground truth seen from the rectified reference camera (same centre, rotated by R1).
    r_wc_rect = r_wc @ rect.R1.T
    _, _, gt_height = render_view(scene, rect.K, None, (640, 480), r_wc_rect, c_left)
    n_true = r_wc_rect.T @ np.array([0.0, 0.0, 1.0])

    assert terrain.valid_fraction > 0.6
    assert abs(terrain.altitude - 5.0) < 0.05
    assert math.degrees(math.acos(np.clip(terrain.plane[:3] @ n_true, -1, 1))) < 0.5
    assert np.nanmedian(np.abs(terrain.height - gt_height)) < 0.03
