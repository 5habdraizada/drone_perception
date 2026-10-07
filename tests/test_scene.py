import numpy as np
import pytest

from perception.scene import SceneConfig, SceneModel

GH, GW, D = 11, 20, 64


def _class_vectors(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, D))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


CLASSES = _class_vectors(4)  # 0 grass, 1 road, 2 roof, 3 something new


def _frame(layout, heights, rng, noise=0.08):
    """layout: GH × GW class ids -> (appearance, height, slope, roughness)."""
    app = CLASSES[layout] + rng.normal(0, noise / np.sqrt(D), (GH, GW, D))
    height = np.asarray(heights, float)[layout] + rng.normal(0, 0.03, (GH, GW))
    slope = np.abs(rng.normal(3, 1, (GH, GW)))
    rough = np.abs(rng.normal(0.02, 0.005, (GH, GW)))
    return app, height, slope, rough


def _base_layout():
    lay = np.zeros((GH, GW), int)
    lay[:, 12:15] = 1          # a road
    lay[2:6, 3:8] = 2          # a roof
    return lay


HEIGHTS = [0.0, 0.0, 1.5, 0.0]


def _purity(regions, truth):
    """Fraction of cells whose region's majority class is their own class."""
    correct = 0
    for r in np.unique(regions):
        cls = truth[regions == r]
        correct += np.bincount(cls).max()
    return correct / truth.size


def test_regions_follow_the_layout():
    rng = np.random.default_rng(1)
    lay = _base_layout()
    frame = SceneModel().update(0.0, *_frame(lay, HEIGHTS, rng))
    assert _purity(frame.regions, lay) > 0.95
    assert len(np.unique(frame.regions)) == 3


def test_region_ids_are_stable_over_time():
    rng = np.random.default_rng(2)
    model, lay = SceneModel(), _base_layout()
    first = model.update(0.0, *_frame(lay, HEIGHTS, rng)).regions
    for t in range(1, 15):
        regions = model.update(float(t), *_frame(lay, HEIGHTS, rng)).regions
    assert (regions == first).mean() > 0.95


def test_geometry_separates_identical_appearance():
    rng = np.random.default_rng(3)
    lay = _base_layout()
    lay[lay == 2] = 0                         # roof now looks exactly like grass ...
    app, height, slope, rough = _frame(lay, HEIGHTS, rng)
    height[2:6, 3:8] = 1.5                    # ... but stands 1.5 m tall
    frame = SceneModel().update(0.0, app, height, slope, rough)
    truth = _base_layout()
    assert _purity(frame.regions, truth) > 0.95
    assert frame.regions[3, 5] != frame.regions[9, 1]


def test_new_object_is_novel_then_becomes_normal():
    rng = np.random.default_rng(4)
    model, lay = SceneModel(), _base_layout()
    for t in range(20):
        model.update(float(t), *_frame(lay, HEIGHTS, rng))

    odd = lay.copy()
    odd[7:9, 16:18] = 3                       # a small object never seen before
    frame = model.update(20.0, *_frame(odd, HEIGHTS, rng))
    assert np.nanmin(frame.novelty[7:9, 16:18]) > 3
    assert np.nanmedian(frame.novelty[odd != 3]) < 1

    for t in range(21, 400):                  # seen constantly, it stops being unusual
        frame = model.update(float(t), *_frame(odd, HEIGHTS, rng))
    assert np.nanmedian(frame.novelty[7:9, 16:18]) < 3


def test_contrast_flags_a_single_odd_cell_without_history():
    rng = np.random.default_rng(9)
    lay = np.zeros((GH, GW), int)
    lay[6, 10] = 3
    frame = SceneModel().update(0.0, *_frame(lay, HEIGHTS, rng))
    assert np.unravel_index(np.nanargmax(frame.contrast), frame.contrast.shape) == (6, 10)


def test_cells_without_depth_get_regions_but_no_novelty():
    rng = np.random.default_rng(5)
    model, lay = SceneModel(), _base_layout()
    for t in range(SceneConfig().warmup_frames + 2):
        app, height, slope, rough = _frame(lay, HEIGHTS, rng)
        height[:, :2] = np.nan                # e.g. the stereo matcher's blind strip
        frame = model.update(float(t), app, height, slope, rough)
    assert (frame.regions[:, :2] >= 0).all()
    assert np.isnan(frame.novelty[:, :2]).all() and np.isfinite(frame.novelty[:, 2:]).all()
    assert not frame.geometry_valid[:, :2].any()


def test_novelty_needs_warmup():
    rng = np.random.default_rng(6)
    frame = SceneModel().update(0.0, *_frame(_base_layout(), HEIGHTS, rng))
    assert np.isnan(frame.novelty).all()


def test_region_count_is_capped():
    rng = np.random.default_rng(7)
    model = SceneModel(SceneConfig(max_regions=5))
    for t in range(30):
        noise = rng.integers(0, 4, (GH, GW))   # chaotic scenes keep trying to add regions
        model.update(float(t), *_frame(noise, HEIGHTS, rng, noise=0.5))
    assert len(model.regions) <= 5


@pytest.mark.slow
def test_real_backbone_separates_roof_from_ground():
    # Full chain on the synthetic scene; needs models/resnet18.onnx from tools/export_backbone.py.
    from pathlib import Path
    model_path = Path("models/resnet18.onnx")
    if not model_path.exists():
        pytest.skip("models/resnet18.onnx not exported")
    pytest.importorskip("onnxruntime")
    from perception.backbone import Backbone
    from perception.terrain import StereoRig, TerrainConfig, TerrainEstimator
    from sim.synthetic_stereo import default_scene, render_stereo

    rig = StereoRig(fx=577.0, fy=577.0, cx=320.0, cy=176.0, baseline=0.10)
    left, right, gt = render_stereo(default_scene(), rig, size=(640, 352), altitude=5.0)
    terrain = TerrainEstimator(rig, TerrainConfig.for_width(640)).process(left, right)
    frame = SceneModel(backbone=Backbone(model_path)).process(0.0, left, terrain)

    c = 32
    h = gt["height"].reshape(11, c, 20, c).transpose(0, 2, 1, 3).reshape(11, 20, -1)
    roof = (h > 1.3).all(axis=-1)
    ground = (np.abs(h) < 0.05).all(axis=-1) & frame.geometry_valid
    roof_ids = np.unique(frame.regions[roof])
    assert roof.sum() >= 4 and ground.sum() > 50
    assert not np.isin(frame.regions[ground], roof_ids).mean() > 0.05  # roof regions stay off the ground


def test_region_info_reports_heights():
    rng = np.random.default_rng(8)
    frame = SceneModel().update(0.0, *_frame(_base_layout(), HEIGHTS, rng))
    roof = frame.regions[3, 5]
    assert frame.region_info[int(roof)]["height"] == pytest.approx(1.5, abs=0.1)
    assert sum(v["cells"] for v in frame.region_info.values()) == GH * GW
