"""Pipeline and tools/run_data.py on generated sample data in every supported layout."""
import csv
import json
import shutil

import cv2
import numpy as np
import pytest

from perception.mapping import pose_from_quaternion, quaternion_from_rotation
from perception.pipeline import Pipeline
from sim.scenes import flight_scene
from tools import make_sample_data, run_data

pytestmark = pytest.mark.slow  # renders sample data


@pytest.fixture(scope="module")
def rectified(tmp_path_factory):
    out = tmp_path_factory.mktemp("rect")
    make_sample_data.main(["--out", str(out), "--frames", "3"])
    return out


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    make_sample_data.main(["--out", str(out), "--frames", "2", "--raw"])
    return out


def _map_error(pipe: Pipeline) -> float:
    g = pipe.map
    rr, cc = np.nonzero(np.isfinite(g.height))
    x, y = g.cell_to_world(rr, cc)
    scene = flight_scene()
    truth = scene.sample(x.astype(np.float32), y.astype(np.float32), scene.height, cv2.INTER_NEAREST).ravel()
    return float(np.median(np.abs(g.height[rr, cc] - truth)))


def test_quaternion_round_trip():
    r = cv2.Rodrigues(np.array([0.3, -1.2, 2.5]))[0]
    t = pose_from_quaternion(1.0, 2.0, 3.0, *quaternion_from_rotation(r))
    assert np.allclose(t[:3, :3], r, atol=1e-9) and np.allclose(t[:3, 3], (1, 2, 3))


def test_run_data_on_rectified_images(rectified, tmp_path):
    out = tmp_path / "run"
    run_data.main([str(rectified), "--out", str(out), "--save-every", "1"])
    rows = list(csv.DictReader(open(out / "frames.csv")))
    assert len(rows) == 3
    assert all(abs(float(r["altitude_m"]) - 6.0) < 0.1 for r in rows)
    assert all(r["mapped"] == "1" for r in rows)
    assert (out / "map.png").exists() and (out / "summary.txt").exists()
    assert len(list((out / "frames").glob("*.png"))) == 3
    assert json.loads((out / "anomalies.json").read_text()) == []


def test_raw_images_with_calibration_land_in_the_right_place(raw):
    pipe = Pipeline.from_calibration(raw / "stereo.json")
    poses = run_data.load_poses(raw / "poses.csv")
    for i, _, l_img, r_img in run_data.stereo_frames(raw):
        res = pipe.process(float(i), l_img, r_img, poses[i])
        assert abs(res.terrain.altitude - 6.0) < 0.1 and res.mapped
    assert _map_error(pipe) < 0.05


def test_rectified_images_land_in_the_right_place(rectified):
    pipe = Pipeline(run_data.load_rig(rectified / "rig.json"), (640, 352))
    poses = run_data.load_poses(rectified / "poses.csv")
    for i, _, l_img, r_img in run_data.stereo_frames(rectified):
        pipe.process(float(i), l_img, r_img, poses[i])
    assert _map_error(pipe) < 0.05


def test_layouts_pairs_and_video(rectified, tmp_path):
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    for side in ("left", "right"):
        for p in sorted((rectified / side).glob("*.png")):
            shutil.copy(p, pairs / f"pair_{p.stem}_{side}.png")
    assert len(list(run_data.stereo_frames(pairs))) == 3

    vids = tmp_path / "video"
    vids.mkdir()
    for side in ("left", "right"):
        writer = cv2.VideoWriter(str(vids / f"{side}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 352))
        for p in sorted((rectified / side).glob("*.png")):
            writer.write(cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR))
        writer.release()
    frames = list(run_data.stereo_frames(vids))
    assert len(frames) == 3 and frames[0][2].shape[:2] == (352, 640)


def test_without_poses_there_is_no_map(rectified, tmp_path):
    data = tmp_path / "noposes"
    shutil.copytree(rectified, data)
    (data / "poses.csv").unlink()
    out = tmp_path / "run"
    run_data.main([str(data), "--out", str(out), "--max-frames", "2"])
    assert not (out / "map.png").exists()
    assert "not built" in (out / "summary.txt").read_text()


def test_raw_images_without_calibration_are_refused(raw, tmp_path):
    data = tmp_path / "rawonly"
    shutil.copytree(raw, data)
    (data / "stereo.json").unlink()
    with pytest.raises(SystemExit, match="--calib"):
        run_data.main([str(data), "--out", str(tmp_path / "run")])
