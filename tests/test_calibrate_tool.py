"""End-to-end test of tools/calibrate_stereo.py on rendered checkerboard captures.

Reproduces the real situation: the provided left-camera distortion is overfit,
so the tool has to notice, re-estimate it, and still recover the stereo geometry.
"""
import cv2
import numpy as np
import pytest

from perception.calibration import (BoardSpec, CameraModel, StereoCalibration, check_model,
                                    model_is_usable)
from sim.synthetic_board import BoardRenderer, sample_poses
from tools.calibrate_stereo import load_intrinsics, main, run_calibration

HALF = (960, 540)  # half resolution keeps rendering and detection quick
BOARD = BoardSpec(9, 6, 0.04)

# "True" cameras: the provided matrices with plausible, well-behaved distortion.
TRUE_L = CameraModel.from_values(1718.54, 1720.12, 1054.17, 454.72, (1920, 1080),
                                 [-0.06, 0.10, 0.0002, -0.0001, 0.0]).scaled(HALF)
TRUE_R = CameraModel.from_values(1175.14, 1177.06, 919.18, 602.39, (1920, 1080),
                                 [-0.20, 0.10, 0.0005, -0.0005, -0.02]).scaled(HALF)
R_TRUE = cv2.Rodrigues(np.radians([0.8, -0.5, 0.3]))[0]
CENTRE = np.array([0.10, 0.002, -0.001])          # right camera centre in the left frame
T_TRUE = -R_TRUE @ CENTRE


@pytest.fixture(scope="module")
def captures():
    rng = np.random.default_rng(7)
    rl, rr = BoardRenderer(TRUE_L, BOARD), BoardRenderer(TRUE_R, BOARD)
    pairs = []
    for rvec, tvec in sample_poses(TRUE_L, BOARD, 20, rng, visible_in=[(TRUE_R, R_TRUE, T_TRUE)]):
        rot = cv2.Rodrigues(rvec)[0]
        pairs.append((rl.render(rvec, tvec, rng),
                      rr.render(cv2.Rodrigues(R_TRUE @ rot)[0], R_TRUE @ tvec + T_TRUE, rng)))
    # Board in the wide camera's periphery, where the narrow one can't see all of it.
    mono_r = [rr.render(rv, tv, rng) for rv, tv in
              sample_poses(TRUE_R, BOARD, 12, rng, depth=(0.7, 1.4),
                           invisible_in=[(TRUE_L, R_TRUE.T, -R_TRUE.T @ T_TRUE)])]
    return pairs, mono_r


def test_provided_intrinsics_are_flagged():
    given = load_intrinsics("config/intrinsics.json")
    assert not model_is_usable(given["left"])
    assert model_is_usable(given["right"])
    assert any("folds back" in m for m in check_model(given["left"]))


def test_recovers_geometry_from_overfit_starting_intrinsics(captures):
    pairs, mono_r = captures
    given = load_intrinsics("config/intrinsics.json")
    calib, rect, summary = run_calibration(pairs, [], mono_r, BOARD, given["left"], given["right"],
                                           log=lambda *_: None)
    assert summary["pairs_used"] >= 15
    assert calib.rms < 0.5
    assert abs(calib.baseline - np.linalg.norm(CENTRE)) < 0.002
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(calib.R @ R_TRUE.T)[0])) < 0.1
    # The refitted left lens is valid everywhere and close to the truth.
    assert model_is_usable(calib.left)
    assert abs(calib.left.K[0, 0] - TRUE_L.K[0, 0]) / TRUE_L.K[0, 0] < 0.01
    assert summary["row_alignment_px"]["mean"] < 0.2
    assert not rect.swapped


def test_cli_writes_a_loadable_calibration(captures, tmp_path):
    pairs, mono_r = captures
    folder = tmp_path / "caps"
    folder.mkdir()
    for i, (l_img, r_img) in enumerate(pairs):
        cv2.imwrite(str(folder / f"pair_{i:03d}_left.png"), l_img)
        cv2.imwrite(str(folder / f"pair_{i:03d}_right.png"), r_img)
    for i, img in enumerate(mono_r):
        cv2.imwrite(str(folder / f"mono_{i:03d}_right.png"), img)
    out = tmp_path / "stereo.json"
    main(["calibrate", "--captures", str(folder), "--square", "0.04",
          "--intrinsics", "config/intrinsics.json", "--out", str(out)])
    calib = StereoCalibration.load(out)
    assert abs(calib.baseline - np.linalg.norm(CENTRE)) < 0.002
    assert calib.meta["pairs_used"] >= 15
    assert (tmp_path / "stereo_rectified_check.png").exists()
