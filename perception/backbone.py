"""Appearance features: a pretrained CNN/ViT cut at an intermediate layer, run with ONNX Runtime.

Models come from tools/export_backbone.py (<name>.onnx + <name>.json). Backends: "cpu" (fp32),
"gpu"/"npu" via onnxruntime's QNN provider (the NPU needs an int8 model).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_QNN_LIB = {"gpu": "QnnGpu", "npu": "QnnHtp"}


@dataclass(frozen=True)
class BackboneSpec:
    """Sidecar metadata written next to each exported model."""
    name: str
    input_size: tuple[int, int]          # (w, h) the model was exported for
    stride: int                          # input pixels per output cell
    channels: int
    mean: tuple[float, float, float]     # per-channel, on 0..1 RGB
    std: tuple[float, float, float]
    pad: int = 0                         # cells of reflected border added around the image, cropped after

    @classmethod
    def load(cls, path: str | Path) -> "BackboneSpec":
        d = json.loads(Path(path).read_text())
        return cls(d["name"], tuple(d["input_size"]), int(d["stride"]), int(d["channels"]),
                   tuple(d["mean"]), tuple(d["std"]), int(d.get("pad", 0)))

    @property
    def image_size(self) -> tuple[int, int]:
        """Size the image itself is resized to, inside the padding."""
        p = 2 * self.pad * self.stride
        return self.input_size[0] - p, self.input_size[1] - p

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


@dataclass
class FeatureGrid:
    features: np.ndarray        # gh × gw × C float32
    image_size: tuple[int, int]  # (w, h) of the image that was passed in

    @property
    def shape(self) -> tuple[int, int]:
        return self.features.shape[:2]

    @property
    def cell_px(self) -> tuple[float, float]:
        """Size of one cell in the input image's pixels (x, y)."""
        gh, gw = self.shape
        return self.image_size[0] / gw, self.image_size[1] / gh

    def resample(self, shape: tuple[int, int]) -> np.ndarray:
        """Features on another (gh, gw) grid, e.g. the terrain cells, by area averaging."""
        if tuple(shape) == self.shape:
            return self.features
        # Separable weight matrices: cv2.resize can't do area averaging on hundreds of channels.
        ay = _resample_matrix(self.shape[0], shape[0])
        ax = _resample_matrix(self.shape[1], shape[1])
        return np.einsum("ij,jkc,lk->ilc", ay, self.features, ax, optimize=True).astype(np.float32)


def _resample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """n_out × n_in weights: area averaging when shrinking, linear interpolation when enlarging."""
    m = np.zeros((n_out, n_in))
    scale = n_in / n_out
    if n_out <= n_in:
        for i in range(n_out):
            lo, hi = i * scale, (i + 1) * scale
            for j in range(int(lo), min(n_in, int(np.ceil(hi)))):
                m[i, j] = min(hi, j + 1) - max(lo, j)
    else:
        for i in range(n_out):
            pos = min(max((i + 0.5) * scale - 0.5, 0.0), n_in - 1.0)  # pixel-centre alignment
            j = int(pos)
            frac = pos - j
            m[i, j] += 1 - frac
            if frac > 0:
                m[i, j + 1] += frac
    return m / m.sum(axis=1, keepdims=True)


class Backbone:
    def __init__(self, model_path: str | Path, backend: str = "cpu", threads: int = 4,
                 normalize: bool = True):
        """threads: CPU threads for inference. The RB3 Gen 2 has 4 fast cores (1 + 3 Cortex-A78).
        normalize: scale every cell's feature vector to unit length, so comparisons
        are cosine similarities and brightness changes matter less."""
        import onnxruntime as ort  # imported here so the rest of the package works without it

        model_path = Path(model_path)
        self.spec = BackboneSpec.load(model_path.with_suffix(".json"))
        self.normalize = normalize

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = self._providers(backend, ort.get_available_providers())
        self._session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self._input = self._session.get_inputs()[0].name
        self.backend = backend

        mean = np.array(self.spec.mean, np.float32) * 255.0
        std = np.array(self.spec.std, np.float32) * 255.0
        self._scale = (1.0 / std).reshape(1, 1, 3)
        self._offset = (-mean / std).reshape(1, 1, 3)

    def __call__(self, image: np.ndarray) -> FeatureGrid:
        """image: grayscale or BGR uint8, any size; it is resized to the model's input size."""
        x = self.preprocess(image)
        out = self._session.run(None, {self._input: x})[0][0]   # C × gh × gw
        p = self.spec.pad
        if p:
            out = out[:, p:-p, p:-p]  # drop the cells that only saw reflected border
        feats = np.ascontiguousarray(out.transpose(1, 2, 0), dtype=np.float32)
        if self.normalize:
            feats /= np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-6
        return FeatureGrid(feats, (image.shape[1], image.shape[0]))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = self.spec.image_size
        if (rgb.shape[1], rgb.shape[0]) != (w, h):
            interp = cv2.INTER_AREA if rgb.shape[1] > w else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (w, h), interpolation=interp)
        if self.spec.pad:
            # Reflected context keeps the network's own zero padding away from real image cells.
            b = self.spec.pad * self.spec.stride
            rgb = cv2.copyMakeBorder(rgb, b, b, b, b, cv2.BORDER_REFLECT_101)
        x = rgb.astype(np.float32) * self._scale + self._offset
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    @staticmethod
    def _providers(backend: str, available: list[str]) -> list:
        if backend == "cpu":
            return ["CPUExecutionProvider"]
        if backend not in _QNN_LIB:
            raise ValueError(f"unknown backend {backend!r}; use 'cpu', 'gpu' or 'npu'")
        if "QNNExecutionProvider" not in available:
            raise RuntimeError(f"backend {backend!r} needs onnxruntime with the QNN execution provider "
                               f"(available here: {', '.join(available)})")
        lib = _QNN_LIB[backend]
        lib = f"{lib}.dll" if os.name == "nt" else f"lib{lib}.so"
        return [("QNNExecutionProvider", {"backend_path": lib}), "CPUExecutionProvider"]
