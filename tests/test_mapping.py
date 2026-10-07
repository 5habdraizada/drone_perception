"""GridMap tests with exact, hand-built frames: a nadir camera over flat ground with one box."""
from types import SimpleNamespace

import numpy as np
import pytest

from perception.mapping import GridMap, MapConfig, camera_from_body, rectified_camera
from perception.terrain import StereoRig

RIG = StereoRig(fx=577.0, fy=577.0, cx=320.0, cy=176.0, baseline=0.10)
SIZE, GH, GW = (640, 352), 11, 20
ALT = 6.0
BOX = (2.0, 4.0, 1.0, 2.0, 1.5)  # x0, x1, y0, y1, height
R_DOWN = np.diag([1.0, -1.0, -1.0])  # camera x right = world x, camera y down = world -y, looking down


def pose(x, y, alt=ALT):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = R_DOWN, (x, y, alt)
    return t


def frame(x, y, t=0.0, novelty_at=None, z=6.0, box=BOX):
    """Scene frame seen from (x, y, ALT): exact cell heights, region 1 on the box, novelty z at a world point."""
    x0, x1, y0, y1, hb = box
    u = (np.arange(GW) + 0.5) * SIZE[0] / GW
    v = (np.arange(GH) + 0.5) * SIZE[1] / GH
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([(uu - RIG.cx) / RIG.fx, (vv - RIG.cy) / RIG.fy, np.ones_like(uu)], -1) @ R_DOWN.T
    top = np.array([x, y, ALT]) + (ALT - hb) * rays          # where each ray meets the box-top plane
    on_box = (top[..., 0] > x0) & (top[..., 0] < x1) & (top[..., 1] > y0) & (top[..., 1] < y1)
    height = np.where(on_box, hb, 0.0)
    ground = np.array([x, y, ALT]) + (ALT - height)[..., None] * rays
    # Rays that miss the top but would land on ground inside the footprint hit a wall: no data.
    wall = ~on_box & (ground[..., 0] > x0) & (ground[..., 0] < x1) & (ground[..., 1] > y0) & (ground[..., 1] < y1)
    height[wall] = np.nan
    novelty = np.zeros((GH, GW))
    if novelty_at is not None:
        nx, ny = novelty_at
        near = (np.abs(ground[..., 0] - nx) < 0.4) & (np.abs(ground[..., 1] - ny) < 0.4)
        novelty[near] = z
    terrain = SimpleNamespace(plane=np.array([0.0, 0.0, -1.0, ALT]), cell_height=height)
    return SimpleNamespace(t=t, regions=on_box.astype(int), novelty=novelty, terrain=terrain)


def flown_map(xs=np.arange(-2, 7, 0.5), **kw):
    m = GridMap(RIG, SIZE, MapConfig(**kw))
    for i, x in enumerate(xs):
        assert m.update(frame(x, 0.5, t=float(i)), pose(x, 0.5))
    return m


def test_box_lands_in_the_right_place_with_the_right_height():
    m = flown_map()
    h = m.height
    tall = np.nan_to_num(h) > 0.75
    rr, cc = np.nonzero(tall)
    xs, ys = m.cell_to_world(rr, cc)
    # Edges blur by up to one scene cell (~0.25-0.33 m on the ground here) plus a map cell:
    # the map cell on an edge averages box-top and ground observations.
    edge = 0.45
    assert xs.min() == pytest.approx(BOX[0], abs=edge) and xs.max() == pytest.approx(BOX[1], abs=edge)
    assert ys.min() == pytest.approx(BOX[2], abs=edge) and ys.max() == pytest.approx(BOX[3], abs=edge)
    r, c = m.world_to_cell(np.array([2.6, 3.0, 3.4]), np.array([1.4, 1.4, 1.4]))  # box interior
    assert np.allclose(h[r, c], 1.5, atol=0.05)
    r, c = m.world_to_cell(0.0, -0.5)
    assert h[r, c] == pytest.approx(0.0, abs=0.05)


def test_elevation_is_world_height():
    m = flown_map()
    r, c = m.world_to_cell(3.0, 1.5)
    assert m.elevation[r, c] == pytest.approx(1.5, abs=0.05)
    r, c = m.world_to_cell(-1.0, 0.0)
    assert m.elevation[r, c] == pytest.approx(0.0, abs=0.05)


def test_cells_without_height_dont_touch_height_or_elevation():
    m = GridMap(RIG, SIZE)
    f = frame(0, 0)
    f.terrain.cell_height = np.full((GH, GW), np.nan)
    assert m.update(f, pose(0, 0))
    assert (m.seen > 0).any()
    assert np.isnan(m.height).all() and np.isnan(m.elevation).all()


def test_regions_layer():
    m = flown_map()
    r, c = m.world_to_cell(3.0, 1.5)
    assert m.regions[r, c] == 1
    r, c = m.world_to_cell(0.0, -0.5)
    assert m.regions[r, c] == 0


def test_one_off_spike_is_not_confirmed_but_repeated_hits_are():
    m = GridMap(RIG, SIZE)
    m.update(frame(0, 0, novelty_at=(1.0, 0.5)), pose(0, 0))
    for i in range(1, 6):
        m.update(frame(0, 0, t=i), pose(0, 0))
    assert m.anomalies() == []

    m = GridMap(RIG, SIZE)
    for i in range(3):
        m.update(frame(i * 0.3, 0, t=i, novelty_at=(1.0, 0.5)), pose(i * 0.3, 0))
    found = m.anomalies()
    assert len(found) == 1
    assert found[0]["x"] == pytest.approx(1.0, abs=0.3) and found[0]["y"] == pytest.approx(0.5, abs=0.3)
    assert found[0]["peak_novelty"] == pytest.approx(6.0)


def test_confirmed_anomaly_survives_getting_used_to_it():
    m = GridMap(RIG, SIZE)
    for i in range(3):
        m.update(frame(0, 0, t=i, novelty_at=(1.0, 0.5)), pose(0, 0))
    for i in range(3, 8):  # the scene model has learned it; novelty back to normal
        m.update(frame(0, 0, t=i), pose(0, 0))
    assert len(m.anomalies()) == 1


def test_map_grows_and_keeps_what_it_had():
    m = flown_map(initial_size=12.0)
    before = m.shape
    m.update(frame(40.0, -30.0, t=99.0), pose(40.0, -30.0))
    assert m.shape[0] > before[0] and m.shape[1] > before[1]
    r, c = m.world_to_cell(3.0, 1.5)
    assert m.height[r, c] == pytest.approx(1.5, abs=0.05)
    r, c = m.world_to_cell(40.0, -30.0)
    assert m.seen[r, c] > 0


def test_frontier_borders_the_explored_strip():
    m = flown_map()
    f = m.frontier()
    assert f.any()
    assert not (f & (m.seen > 0)).any()
    rr, cc = np.nonzero(f)
    _, ys = m.cell_to_world(rr, cc)
    # The camera sees about +/- 1.8 m across track around y = 0.5.
    assert ys.max() == pytest.approx(0.5 + 1.83, abs=0.4) and ys.min() == pytest.approx(0.5 - 1.83, abs=0.4)


def test_clearance():
    m = flown_map()
    assert m.clearance(3.0, 1.5, 0.5) == pytest.approx(1.5, abs=0.05)
    assert m.clearance(-1.0, -0.5, 0.5) == pytest.approx(0.0, abs=0.05)
    assert np.isnan(m.clearance(100.0, 100.0, 1.0))


def test_uncertain_poses_count_less():
    m = GridMap(RIG, SIZE)
    for i in range(3):
        m.update(frame(3.0, 1.5, t=i), pose(3.0, 1.5))
    flat = (0.0, 0.0, 0.0, 0.0, 0.0)  # same place, box missing, from a badly localised pose
    for i in range(3, 6):
        m.update(frame(3.0, 1.5, t=i, box=flat), pose(3.0, 1.5), pose_cov=np.diag([4.0, 4.0, 1, 0.1, 0.1, 0.1]))
    r, c = m.world_to_cell(3.0, 1.5)
    assert m.height[r, c] > 1.3


def test_frame_without_ground_plane_is_skipped():
    m = GridMap(RIG, SIZE)
    f = frame(0, 0)
    f.terrain.plane = None
    assert not m.update(f, pose(0, 0))


def test_pose_helpers():
    t_wb = pose(1.0, 2.0)
    mount = np.eye(4)
    mount[:3, 3] = (0.05, 0.0, 0.0)
    t_wc = camera_from_body(t_wb, mount)
    assert np.allclose(t_wc[:3, 3], (1.05, 2.0, ALT))
    r1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    t_rect = rectified_camera(t_wc, r1)
    # A point on the rectified optical axis maps to the same world point as R1^T of it in the raw frame.
    p_rect = np.array([0.0, 0.0, 3.0, 1.0])
    assert np.allclose(t_rect @ p_rect, t_wc @ np.append(r1.T @ p_rect[:3], 1.0))


def test_render_shapes():
    m = flown_map()
    full = m.render(crop=False)
    assert full.shape == (m.shape[0], 4 * m.shape[1] + 3 * 4, 3)
    cropped = m.render()
    assert cropped.shape[0] < full.shape[0] and cropped.shape[1] < full.shape[1]
    assert m.render(scale=3).shape[0] == 3 * cropped.shape[0]
