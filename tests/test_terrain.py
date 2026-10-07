import math

import numpy as np
import pytest

from perception.terrain import StereoRig, TerrainConfig, TerrainEstimator, box_sum
from sim.synthetic_stereo import Block, Scene, default_scene, render_stereo

RIG = StereoRig(fx=300.0, fy=300.0, cx=160.0, cy=120.0, baseline=0.10)
ALTITUDE = 5.0


def _angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


@pytest.fixture(scope="module")
def tilted():
    left, right, gt = render_stereo(default_scene(), RIG, altitude=ALTITUDE, roll_deg=4, pitch_deg=-3)
    return TerrainEstimator(RIG).process(left, right), gt


@pytest.fixture(scope="module")
def big_roof():
    # A 1.2 m roof covering most of the view; the ground shows only on one side.
    scene = Scene(blocks=[Block(x=-0.6, y=0.0, w=4.0, l=5.0, h=1.2)])
    return render_stereo(scene, RIG, altitude=ALTITUDE)


def _mask(gt, lo, hi, terrain):
    return (gt["height"] > lo) & (gt["height"] < hi) & np.isfinite(terrain.height)


def test_coverage(tilted):
    terrain, _ = tilted
    assert terrain.valid_fraction > 0.6


def test_ground_plane(tilted):
    terrain, gt = tilted
    assert terrain.plane is not None
    assert abs(terrain.altitude - ALTITUDE) < 0.1
    assert _angle_deg(terrain.plane[:3], gt["plane"][:3]) < 1.5


@pytest.mark.parametrize("lo, hi, truth", [
    (-0.05, 0.05, 0.0),   # ground
    (1.3, 1.7, 1.5),      # tall structure
    (0.6, 1.0, 0.8),      # short structure
    (-1.0, -0.6, -0.8),   # pit
])
def test_heights(tilted, lo, hi, truth):
    terrain, gt = tilted
    sel = _mask(gt, lo, hi, terrain)
    assert sel.sum() > 500
    assert abs(np.median(terrain.height[sel]) - truth) < 0.1


def test_slope_separates_walls_from_flat(tilted):
    terrain, gt = tilted
    c = 16
    g = gt["height"][:240, :320].reshape(15, c, 20, c).transpose(0, 2, 1, 3).reshape(15, 20, -1)
    spread = g.max(-1) - g.min(-1)
    # Flat = the whole pooled neighbourhood is flat in ground truth.
    flat = box_sum((spread > 0.02).astype(float), 3) == 0
    wall = spread > 0.5
    assert np.nanpercentile(terrain.slope_deg[flat], 95) < 10
    assert np.nanmedian(terrain.slope_deg[wall]) > 45
    assert np.nanmedian(terrain.roughness[flat]) < 0.05


def test_cell_height_tracks_structures(tilted):
    terrain, _ = tilted
    assert np.nanmax(terrain.cell_height) > 1.2
    assert np.nanmin(terrain.cell_height) < -0.5


def test_big_roof_without_prior_is_mistaken_for_ground(big_roof):
    # Documents the failure mode the prior exists for.
    left, right, _ = big_roof
    terrain = TerrainEstimator(RIG).process(left, right)
    assert abs(terrain.altitude - (ALTITUDE - 1.2)) < 0.2


def test_prior_keeps_true_ground_under_big_roof(big_roof):
    left, right, gt = big_roof
    terrain = TerrainEstimator(RIG).process(left, right, plane_prior=gt["plane"])
    assert abs(terrain.altitude - ALTITUDE) < 0.1
    roof = (gt["height"] > 1.1) & np.isfinite(terrain.height)
    assert abs(np.median(terrain.height[roof]) - 1.2) < 0.1


def test_prior_follows_real_attitude_and_altitude_change():
    # Tracking must not lag when the drone genuinely tilts and climbs between frames.
    scene = default_scene()
    est = TerrainEstimator(RIG)
    first = est.process(*render_stereo(scene, RIG, altitude=ALTITUDE)[:2])
    left, right, gt = render_stereo(scene, RIG, altitude=ALTITUDE + 0.3, roll_deg=6, pitch_deg=-4)
    second = est.process(left, right, plane_prior=first.plane)
    assert abs(second.altitude - (ALTITUDE + 0.3)) < 0.1
    assert _angle_deg(second.plane[:3], gt["plane"][:3]) < 1.5


@pytest.mark.slow
def test_640_resolution_is_more_accurate():
    # RB3 target: same scene at 640 × 480 with pixel settings scaled via for_width.
    rig = StereoRig(fx=600.0, fy=600.0, cx=320.0, cy=240.0, baseline=0.10)
    left, right, gt = render_stereo(default_scene(), rig, size=(640, 480),
                                    altitude=ALTITUDE, roll_deg=4, pitch_deg=-3)
    terrain = TerrainEstimator(rig, TerrainConfig.for_width(640)).process(left, right)
    assert terrain.cell_height.shape == (15, 20)  # same grid as at 320 px
    assert abs(terrain.altitude - ALTITUDE) < 0.05
    assert _angle_deg(terrain.plane[:3], gt["plane"][:3]) < 0.5
    assert np.nanmedian(np.abs(terrain.height - gt["height"])) < 0.02


def test_for_width_scales_pixel_settings():
    cfg = TerrainConfig.for_width(640, matcher="sgbm")
    assert (cfg.cell_px, cfg.block_size, cfg.edge_window) == (32, 31, 19)
    assert cfg.matcher == "sgbm"
    assert TerrainConfig.for_width(320) == TerrainConfig()


def test_sgbm_is_denser_than_bm(tilted):
    # SGBM is opt-in: it pixel-locks on gentle slopes, which skews the ground
    # plane and every height measured from it (see TerrainConfig.matcher).
    # What it does buy is coverage.
    bm_terrain, _ = tilted
    left, right, _ = render_stereo(default_scene(), RIG, altitude=ALTITUDE, roll_deg=4, pitch_deg=-3)
    terrain = TerrainEstimator(RIG, TerrainConfig(matcher="sgbm")).process(left, right)
    assert terrain.plane is not None
    assert terrain.valid_fraction > bm_terrain.valid_fraction + 0.05


def test_unknown_matcher_is_rejected():
    with pytest.raises(ValueError):
        TerrainEstimator(RIG, TerrainConfig(matcher="nope"))


def test_untextured_input_returns_no_plane():
    blank = np.full((240, 320), 128, np.uint8)
    terrain = TerrainEstimator(RIG).process(blank, blank)
    assert terrain.plane is None
    assert math.isnan(terrain.altitude)
    assert np.isnan(terrain.height).all()


def test_rig_from_q_matrix():
    q = np.array([[1, 0, 0, -160.0], [0, 1, 0, -120.0], [0, 0, 0, 300.0], [0, 0, 10.0, 0]])
    rig = StereoRig.from_Q(q)
    assert (rig.fx, rig.cx, rig.cy) == (300.0, 160.0, 120.0)
    assert rig.baseline == pytest.approx(0.1)
