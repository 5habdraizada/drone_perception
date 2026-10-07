"""Closed-loop mission tests: oracle perception -> map -> planner -> simulated drone."""
import math

import numpy as np
import pytest

from perception.decision import DecisionConfig, MissionPlanner
from perception.mapping import GridMap
from perception.terrain import StereoRig
from sim.mission import DroneSim, run_mission
from sim.oracle_perception import OraclePerception
from sim.synthetic_stereo import Block, Scene

RIG = StereoRig(fx=577.0, fy=577.0, cx=320.0, cy=176.0, baseline=0.10)
SIZE = (640, 352)
AREA = (-8.0, 8.0, -6.0, 6.0)


def _scene(*blocks):
    return Scene(extent=30.0, res=0.05, blocks=list(blocks))  # coarse: the oracle doesn't use textures


def _run(scene, area=AREA, battery=1.0, sigma_growth=0.0, false_alarms=0.0, **cfg):
    perception = OraclePerception(scene, RIG, SIZE, false_alarm_rate=false_alarms, seed=1)
    grid = GridMap(RIG, SIZE)
    planner = MissionPlanner(home=(0.0, 0.0), cfg=DecisionConfig(area=area, **cfg))
    drone = DroneSim(0.0, 0.0, battery=battery)
    return run_mission(perception, grid, planner, drone, sigma_growth=sigma_growth)


def _modes(result):
    return [s["mode"] for s in result.trace]


def test_explores_the_whole_area_then_lands_at_home():
    res = _run(_scene())
    assert res.landed
    last = res.trace[-1]
    assert math.hypot(last["x"], last["y"]) < 1.0
    covered = (res.grid.seen > 0) & res.grid.area_mask(*AREA)
    assert covered.sum() / res.grid.area_mask(*AREA).sum() > 0.95
    assert "investigate" not in _modes(res)
    assert any("covered" in s["reason"] for s in res.trace)


def test_finds_and_investigates_an_unusual_object():
    odd = Block(5.0, 3.0, 0.8, 0.8, 0.5, pattern="stripes")
    res = _run(_scene(odd))
    assert res.landed
    assert len(res.planner.report) == 1
    found = res.planner.report[0]
    assert math.hypot(found["x"] - odd.x, found["y"] - odd.y) < 0.75
    assert found["height"] == pytest.approx(0.5, abs=0.15)
    holding = [s for s in res.trace if s["mode"] == "investigate" and math.hypot(s["x"] - odd.x, s["y"] - odd.y) < 1.0]
    assert min(s["agl"] for s in holding) == pytest.approx(DecisionConfig().investigate_agl, abs=0.3)


def test_one_frame_false_alarms_are_not_investigated():
    res = _run(_scene(), false_alarms=0.3)
    assert res.landed
    assert res.planner.report == []
    assert "verify" in _modes(res)  # it does go back to look, and the map doesn't confirm them


BIG = (-15.0, 15.0, -12.0, 12.0)  # ~90 s mission, enough for battery and drift to matter


def test_low_battery_brings_it_home():
    res = _run(_scene(), area=BIG, battery=0.22)
    assert res.landed
    assert any(s["mode"] == "return" and "battery" in s["reason"] for s in res.trace)
    last = res.trace[-1]
    assert math.hypot(last["x"], last["y"]) < 1.0 and last["battery"] > 0.05


def test_growing_position_uncertainty_brings_it_home():
    res = _run(_scene(), area=BIG, sigma_growth=0.06)
    assert res.landed
    assert any(s["mode"] == "return" and "uncertainty" in s["reason"] for s in res.trace)


def test_waypoint_altitude_clears_mapped_obstacles_and_unknown_space():
    roof = Block(4.0, 0.0, 3.0, 3.0, 2.5)
    perception = OraclePerception(_scene(roof), RIG, SIZE)
    grid = GridMap(RIG, SIZE)
    for x in np.arange(0.0, 8.0, 0.5):
        grid.update(perception.frame(x, 0.0, 8.0, x), perception.pose(x, 0.0, 8.0))
    planner = MissionPlanner(home=(0.0, 0.0), cfg=DecisionConfig(investigate_agl=1.0))
    assert planner.path_clearance(grid, 0.0, 0.0, 8.0, 0.0) == pytest.approx(2.5, abs=0.05)
    cmd = planner._go(grid, 0.0, 0.0, "investigate", 8.0, 0.0, 1.0, "test")
    assert cmd.agl == pytest.approx(2.5 + DecisionConfig().min_clearance, abs=0.05)
    far = planner._go(grid, 0.0, 0.0, "explore", 0.0, 40.0, 1.0, "test")  # across unmapped ground
    assert far.agl == pytest.approx(DecisionConfig().unknown_height + DecisionConfig().min_clearance)


def test_explores_without_dithering():
    # Explore targets move on as the camera maps them (that's fine); what must not happen
    # is the drone turning back and forth.
    res = _run(_scene(), area=BIG)
    tr = res.trace
    steps = [(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(tr, tr[1:])]
    heads = [math.atan2(dy, dx) for dx, dy in steps if math.hypot(dx, dy) > 0.5]
    turn = lambda a, b: abs((b - a + math.pi) % (2 * math.pi) - math.pi)
    assert sum(turn(a, b) > math.radians(150) for a, b in zip(heads, heads[1:])) <= 2
    flown = sum(math.hypot(dx, dy) for dx, dy in steps)
    area = (BIG[1] - BIG[0]) * (BIG[3] - BIG[2])
    assert flown < 3.0 * area / 7.3  # within 3x an ideal sweep (7.3 m swath) including the way home
