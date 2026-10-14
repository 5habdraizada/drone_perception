"""Mission steering: reply parsing, the async worker, best-view crops, and closed-loop missions."""
import math
import time

import numpy as np
import pytest

from perception.advisor import AsyncAdvisor, FunctionAdvisor, Snapshots, parse_reply
from perception.decision import DecisionConfig, MissionPlanner
from perception.mapping import GridMap
from perception.terrain import StereoRig
from sim.mission import DroneSim, run_mission
from sim.oracle_perception import OraclePerception
from sim.synthetic_stereo import Block, Scene

RIG = StereoRig(fx=577.0, fy=577.0, cx=320.0, cy=176.0, baseline=0.10)
SIZE = (640, 352)
AREA = (-8.0, 8.0, -6.0, 6.0)
VEHICLE = Block(5.0, 3.0, 0.8, 0.8, 0.5, pattern="stripes")     # matches the brief
TARP = Block(-5.0, -3.0, 0.8, 0.8, 0.0, pattern="stripes")      # unusual, but not what we're looking for


def test_parse_reply_handles_messy_model_output():
    a = parse_reply('Sure! {"label": "red car", "relevance": 0.9, "investigate": true, "reason": "a vehicle"} done',
                    "m")
    assert (a.label, a.relevance, a.investigate) == ("red car", 0.9, True)
    b = parse_reply('{"label": "tree", "relevance": 7, "investigate": "no", "reason": ""}', "m")
    assert b.relevance == 1.0 and b.investigate is False
    # Unparseable: unsure, and still worth a look; a model failure must never hide a target.
    c = parse_reply("I cannot tell from this image.", "m")
    assert c.investigate and c.relevance == 0.5


def test_async_advisor_answers_once_in_the_background_and_survives_errors():
    calls = []

    def slow(image, brief):
        calls.append(brief)
        time.sleep(0.05)
        return "thing", 0.8, True, "ok"

    adv = AsyncAdvisor(FunctionAdvisor(slow), brief="find things")
    adv.request("a", np.zeros((8, 8)))
    adv.request("a", np.zeros((8, 8)))            # duplicate request is ignored
    assert adv.result("a") is None                # not blocking: no answer yet
    adv.wait()
    assert adv.result("a").relevance == 0.8 and calls == ["find things"]

    def broken(image, brief):
        raise RuntimeError("model crashed")

    bad = AsyncAdvisor(FunctionAdvisor(broken), brief="x")
    bad.request("b", np.zeros((8, 8)))
    bad.wait()
    assert bad.result("b").investigate and "model crashed" in bad.result("b").reason


def test_snapshots_crop_the_best_view_of_a_world_point():
    k = np.array([[577.0, 0, 320], [0, 577.0, 176], [0, 0, 1]])
    down = np.diag([1.0, -1.0, -1.0])
    snaps = Snapshots(k)
    for cx in (-3.0, 0.0, 3.0):                   # three frames; the middle one sees (0, 0) dead centre
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = down, (cx, 0.0, 10.0)
        img = np.zeros((352, 640), np.uint8)
        u = int(320 + (0.0 - cx) * 577 / 10)      # where the world origin lands in this frame
        img[171:181, u - 5:u + 5] = 255 if cx == 0.0 else 100
        snaps.add(img, pose)
    crop = snaps.crop(0.0, 0.0, size_m=2.0, out_px=64)
    assert crop.shape == (64, 64)
    assert crop[30:34, 30:34].mean() > 200        # the bright marker (~5 px here) from the centred frame
    assert snaps.crop(500.0, 0.0) is None


def _mission(advisor_fn, battery=1.0, **cfg):
    scene = Scene(extent=30.0, res=0.05, blocks=[VEHICLE, TARP])
    perception = OraclePerception(scene, RIG, SIZE, seed=1)
    adv = AsyncAdvisor(FunctionAdvisor(advisor_fn), brief="Find vehicles. Ignore tarps and debris.")
    # Crops: stand-ins that encode which object is there (a real run uses Snapshots.crop).
    crops = lambda a: np.full((8, 8), 1.0 if math.hypot(a["x"] - VEHICLE.x, a["y"] - VEHICLE.y) < 1.5 else 0.0)
    planner = MissionPlanner(home=(0.0, 0.0), cfg=DecisionConfig(area=AREA, **cfg), advisor=adv, crops=crops)
    return run_mission(perception, GridMap(RIG, SIZE), planner, DroneSim(0.0, 0.0, battery=battery))


def _vehicle_judge(image, brief):
    if image.mean() > 0.5:
        return "vehicle", 0.9, True, "looks like a vehicle"
    return "tarp", 0.05, False, "a tarp: not in the brief"


def test_advisor_steers_the_mission_to_what_the_brief_asks_for():
    res = _mission(_vehicle_judge)
    assert res.landed
    found = res.planner.report
    assert len(found) == 1 and math.hypot(found[0]["x"] - VEHICLE.x, found[0]["y"] - VEHICLE.y) < 1.0
    assert found[0]["advice"]["label"] == "vehicle"
    gone = res.planner.dismissed
    assert len(gone) == 1 and math.hypot(gone[0]["x"] - TARP.x, gone[0]["y"] - TARP.y) < 1.0
    assert "not in the brief" in gone[0]["advice"]["reason"]


def test_without_an_advisor_both_objects_are_investigated():
    scene = Scene(extent=30.0, res=0.05, blocks=[VEHICLE, TARP])
    perception = OraclePerception(scene, RIG, SIZE, seed=1)
    planner = MissionPlanner(home=(0.0, 0.0), cfg=DecisionConfig(area=AREA))
    res = run_mission(perception, GridMap(RIG, SIZE), planner, DroneSim(0.0, 0.0))
    assert len(res.planner.report) == 2


def test_advisor_cannot_override_safety():
    # An advisor that wants everything investigated can't stop a low-battery return.
    res = _mission(lambda img, brief: ("anything", 1.0, True, "investigate everything"),
                   battery=0.2, battery_per_m=0.004)
    assert res.landed
    assert any(s["mode"] == "return" and "battery" in s["reason"] for s in res.trace)
    last = res.trace[-1]
    assert math.hypot(last["x"], last["y"]) < 1.0
