"""sim/real_world.py on a tiny hand-made area: orientation, ground filling, truth heights."""
import json

import cv2
import numpy as np
import pytest

from sim.real_world import load_area


@pytest.fixture
def area_dir(tmp_path):
    n = 40                                    # 20 m square at 0.5 m
    dtm = np.full((n, n), 1.0, np.float32)    # ground at 1 m NAP, north-up rasters
    dsm = dtm.copy()
    dsm[2:10, 28:38] = 13.0                   # a 12 m building in the north-east corner
    dtm[2:10, 28:38] = 3.4e38                 # the ground model has no data under it
    dsm[30:34, 2:6] = 3.4e38                  # a hole in the surface model (water)
    cv2.imwrite(str(tmp_path / "dsm.tif"), dsm)
    cv2.imwrite(str(tmp_path / "dtm.tif"), dtm)
    ortho = np.full((200, 200), 90, np.uint8)  # 10 cm photo
    ortho[20:100, 140:190] = 220              # bright roof, same corner
    cv2.imwrite(str(tmp_path / "ortho.png"), ortho)
    (tmp_path / "area.json").write_text(json.dumps({"bbox": [0, 0, 20, 20], "dem_resolution": 0.5}))
    return tmp_path


def test_building_lands_north_east_with_the_right_height(area_dir):
    area = load_area(area_dir)
    assert area.extent == 20 and area.scene.res == pytest.approx(0.1)
    # Local coordinates: origin at the centre, x east, y north.
    assert area.height_at(6.0, 7.0) == pytest.approx(12.0)       # on the building
    assert area.height_at(-6.0, -7.0) == pytest.approx(0.0)      # south-west ground
    assert np.isnan(area.height_at(50.0, 0.0))
    # The renderer's heightfield agrees: world z relative to street level.
    s = area.scene
    z = s.sample(np.float32([[6.0]]), np.float32([[7.0]]), s.height, cv2.INTER_NEAREST)
    assert float(z[0, 0]) == pytest.approx(12.0)
    tex = s.sample(np.float32([[6.0]]), np.float32([[7.0]]), s.texture, cv2.INTER_NEAREST)
    assert float(tex[0, 0]) == pytest.approx(220 / 255, abs=0.01)


def test_holes_are_filled(area_dir):
    area = load_area(area_dir)
    assert np.isfinite(area.truth_height).all()
    assert area.height_at(-8.0, -6.0) == pytest.approx(0.0)       # the water hole reads as ground
