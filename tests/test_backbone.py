"""Backbone plumbing, tested with a tiny hand-built ONNX model (no downloaded weights)."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from perception.backbone import Backbone, BackboneSpec  # noqa: E402

IN_W, IN_H, STRIDE, CH = 320, 176, 16, 8
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """A single 16×16 stride-16 convolution: patchify into an 8-channel 20 × 11 grid."""
    rng = np.random.default_rng(0)
    weight = rng.normal(0, 0.05, (CH, 3, STRIDE, STRIDE)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["image", "w"], ["features"], kernel_shape=[STRIDE, STRIDE],
                          strides=[STRIDE, STRIDE])],
        "patchify",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, IN_H, IN_W])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, CH, IN_H // STRIDE, IN_W // STRIDE])],
        [numpy_helper.from_array(weight, "w")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    onnx.save(m, str(path))
    BackboneSpec("tiny", (IN_W, IN_H), STRIDE, CH, MEAN, STD).save(path.with_suffix(".json"))
    return path, weight


def _frame(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (352, 640), dtype=np.uint8)


def test_grid_matches_terrain_cells(model):
    grid = Backbone(model[0])(_frame())
    assert grid.shape == (11, 20)
    assert grid.features.shape == (11, 20, CH) and grid.features.dtype == np.float32
    assert grid.cell_px == (32.0, 32.0)  # one feature per 32 px terrain cell


def test_features_are_unit_length_when_normalised(model):
    feats = Backbone(model[0])(_frame()).features
    assert np.allclose(np.linalg.norm(feats, axis=-1), 1.0, atol=1e-4)
    raw = Backbone(model[0], normalize=False)(_frame()).features
    assert not np.allclose(np.linalg.norm(raw, axis=-1), 1.0)


def test_output_matches_reference_convolution(model):
    path, weight = model
    bb = Backbone(path, normalize=False)
    img = _frame()
    x = bb.preprocess(img)[0]                                   # 3 × H × W
    patches = x.reshape(3, IN_H // STRIDE, STRIDE, IN_W // STRIDE, STRIDE)
    ref = np.einsum("cyixj,kcij->yxk", patches, weight)
    assert np.allclose(bb(img).features, ref, atol=1e-4)


def test_preprocessing_applies_mean_and_std(model):
    x = Backbone(model[0]).preprocess(np.full((176, 320), 128, np.uint8))
    expected = (128 / 255 - np.array(MEAN)) / np.array(STD)
    assert x.shape == (1, 3, IN_H, IN_W)
    assert np.allclose(x[0, :, 5, 5], expected, atol=1e-5)


def test_gray_and_bgr_inputs_agree(model):
    bb = Backbone(model[0])
    gray = _frame()
    bgr = np.repeat(gray[..., None], 3, axis=2)
    assert np.allclose(bb(gray).features, bb(bgr).features, atol=1e-5)


def test_resample_to_another_grid(model):
    grid = Backbone(model[0])(_frame())
    assert grid.resample((11, 20)) is grid.features
    assert grid.resample((5, 10)).shape == (5, 10, CH)


def test_resample_handles_more_than_512_channels():
    from perception.backbone import FeatureGrid
    grid = FeatureGrid(np.random.default_rng(0).random((11, 20, 960), dtype=np.float32), (640, 352))
    out = grid.resample((22, 40))
    assert out.shape == (22, 40, 960)
    assert np.allclose(out[::2, ::2].mean(), grid.features.mean(), atol=0.02)


def test_unknown_backend_is_rejected(model):
    with pytest.raises(ValueError):
        Backbone(model[0], backend="tpu")


def test_npu_without_qnn_provider_explains_itself(model):
    import onnxruntime as ort
    if "QNNExecutionProvider" in ort.get_available_providers():
        pytest.skip("QNN provider present")
    with pytest.raises(RuntimeError, match="QNN"):
        Backbone(model[0], backend="npu")
