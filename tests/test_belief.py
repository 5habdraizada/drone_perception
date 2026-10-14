"""Belief map: what clear views, canopy, roofs, hits and depressions do to it."""
import numpy as np
import pytest

from perception.belief import BeliefConfig, BeliefMap
from perception.mapping import GridMap, MapConfig
from perception.terrain import StereoRig
from sim.oracle_perception import OraclePerception
from sim.synthetic_stereo import Block, Scene

RIG = StereoRig(fx=577.0, fy=577.0, cx=320.0, cy=176.0, baseline=0.10)
SIZE = (640, 352)
TREES = Block(6.0, 0.0, 6.0, 6.0, 5.0, pattern="canopy")
ROOF = Block(-6.0, 0.0, 6.0, 6.0, 5.0)
HIDDEN = Block(6.0, 0.0, 1.0, 1.0, 0.0, pattern="hidden")


def _setup(*blocks, res=0.5):
    oracle = OraclePerception(Scene(extent=60.0, res=0.1, blocks=list(blocks)), RIG, SIZE, seed=2)
    grid = GridMap(RIG, SIZE, MapConfig(resolution=res))
    grid.ensure_area(-20, 20, -20, 20)
    return oracle, grid, BeliefMap(grid, RIG, SIZE)


def _fly(oracle, grid, belief, xs, agl, y=0.0):
    for i, x in enumerate(xs):
        f, pose = oracle.frame(x, y, agl, float(i)), oracle.pose(x, y, agl)
        grid.update(f, pose)
        belief.observe(f, pose)


def _at(grid, layer, x, y):
    r, c = grid.world_to_cell(x, y)
    return layer[r, c]


def test_clear_views_rule_out_open_ground_but_not_under_canopy_and_roofs_are_unsearchable():
    oracle, grid, belief = _setup(TREES, ROOF)
    prior = BeliefConfig().object_prior
    _fly(oracle, grid, belief, np.arange(-8, 9, 2.0), agl=20)
    p = belief.p_object
    assert _at(grid, p, 0.0, 5.0) < prior / 3                      # open ground, seen clearly
    assert prior / 3 < _at(grid, p, 6.0, 0.0) < prior              # under canopy: barely ruled out
    assert not _at(grid, belief.searchable, -6.0, 0.0)            # roof
    assert _at(grid, p, -6.0, 0.0) == 0.0


def test_entropy_falls_as_the_area_is_observed():
    oracle, grid, belief = _setup(TREES)
    area = (-15, 15, -10, 10)
    before = belief.entropy(area)["object"]
    _fly(oracle, grid, belief, np.arange(-10, 11, 2.0), agl=20)
    assert belief.entropy(area)["object"] < 0.6 * before


def test_low_passes_find_what_canopy_hides():
    oracle, grid, belief = _setup(TREES, HIDDEN)
    _fly(oracle, grid, belief, np.arange(0, 12, 1.0), agl=25)       # high: rarely glimpsed
    high = _at(grid, belief.p_object, HIDDEN.x, HIDDEN.y)
    _fly(oracle, grid, belief, np.arange(3, 10, 0.5), agl=8)        # low pass over the trees
    found = belief.detections("object")
    assert any(np.hypot(d["x"] - HIDDEN.x, d["y"] - HIDDEN.y) < 1.5 for d in found)
    assert _at(grid, belief.p_object, HIDDEN.x, HIDDEN.y) > max(high, 0.9)


def test_expected_gain_prefers_unexplored_ground_and_low_passes_over_canopy():
    oracle, grid, belief = _setup(TREES)
    _fly(oracle, grid, belief, np.arange(-12, -2, 2.0), agl=20)     # the west is now well seen
    assert belief.expected_gain(-8, 0, 20) < 0.5 * belief.expected_gain(10, 0, 20)
    _fly(oracle, grid, belief, np.arange(2, 12, 2.0), agl=20)       # trees seen from high up
    # Another high pass re-sees the same canopy gaps: almost nothing left to gain there. A low
    # pass resolves gaps the high one couldn't, so it's where the remaining information is.
    again_high, low = belief.expected_gain(6, 0, 20), belief.expected_gain(6, 0, 8)
    assert again_high < 0.1 * low


def test_occluded_ground_needs_a_viewpoint_from_the_side():
    # The strip just north of the trees is hidden by the canopy edge when looking from overhead.
    oracle, grid, belief = _setup(TREES)
    _fly(oracle, grid, belief, np.arange(0, 13, 2.0), agl=20)
    strip = (6.0, 3.8)
    assert _at(grid, belief.view_quality, *strip) == 0.0              # never actually seen
    overhead, beside = belief.expected_gain(6, 0, 20), belief.expected_gain(6, 9, 20)
    assert beside > 3 * overhead                                     # move north to see it
    _fly(oracle, grid, belief, np.arange(0, 13, 2.0), agl=20, y=9.0)
    assert _at(grid, belief.view_quality, *strip) > 0.3


def test_depressions_raise_subsurface_belief():
    pit = Block(0.0, 4.0, 3.0, 3.0, -1.0)
    oracle, grid, belief = _setup(pit)
    _fly(oracle, grid, belief, np.arange(-4, 5, 1.0), agl=15)
    assert _at(grid, belief.p_subsurface, 0.0, 4.0) > 5 * _at(grid, belief.p_subsurface, 0.0, -6.0)


def test_belief_survives_the_map_growing():
    oracle, grid, belief = _setup(TREES, ROOF)
    _fly(oracle, grid, belief, [0.0], agl=20)
    before = _at(grid, belief.p_object, 0.0, 5.0)
    grid.ensure_area(-80, 80, -80, 80)
    belief._sync()
    assert _at(grid, belief.p_object, 0.0, 5.0) == pytest.approx(before)
    assert _at(grid, belief.p_object, -70.0, -70.0) == pytest.approx(BeliefConfig().object_prior)
