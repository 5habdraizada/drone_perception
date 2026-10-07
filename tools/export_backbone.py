"""Export pretrained feature extractors to ONNX for perception/backbone.py.

Runs on a development machine (needs torch + timm, see requirements-dev.txt);
the drone only needs the resulting models/<name>.onnx and models/<name>.json.

    python -m tools.export_backbone                      # all candidates
    python -m tools.export_backbone mobilenetv3_large dinov2_small

Each network is cut at the layer whose output grid matches the terrain cells
(20 × 11 for a 640 × 352 rectified frame): stride 16 on a 320 × 176 image for
the CNNs, and 14 px patches on a 280 × 154 image for DINOv2. One extra cell of
reflected border goes around the image and is cropped off at run time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perception.backbone import BackboneSpec

GRID = (20, 11)  # terrain cells across a 640 × 352 rectified frame
PAD = 1          # cells of reflected border, cropped off at run time (see perception/backbone.py)

# name: (timm model id, kind, stride)
CANDIDATES = {
    "mobilenetv3_small": ("mobilenetv3_small_100.lamb_in1k", "cnn", 16),
    "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k", "cnn", 16),
    "efficientnet_lite0": ("tf_efficientnet_lite0.in1k", "cnn", 16),
    "resnet18": ("resnet18.a1_in1k", "cnn", 16),
    "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", "vit", 14),
}


def build(model_id: str, kind: str, stride: int, input_size: tuple[int, int]):
    """Return (torch module mapping 1×3×H×W -> 1×C×gh×gw, channels, mean, std)."""
    import timm
    import torch

    w, h = input_size
    if kind == "cnn":
        probe = timm.create_model(model_id, pretrained=False, features_only=True)
        reductions = probe.feature_info.reduction()
        if stride not in reductions:
            raise ValueError(f"{model_id} has no stride-{stride} layer (has {reductions})")
        idx = reductions.index(stride)
        net = timm.create_model(model_id, pretrained=True, features_only=True, out_indices=(idx,))
        channels = net.feature_info.channels()[0]

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[0]
    else:
        net = timm.create_model(model_id, pretrained=True, img_size=(h, w), num_classes=0)
        channels = net.num_features
        prefix = net.num_prefix_tokens
        gh, gw = h // stride, w // stride

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                tokens = self.net.forward_features(x)[:, prefix:]        # B × N × C
                return tokens.reshape(x.shape[0], gh, gw, -1).permute(0, 3, 1, 2)

    cfg = net.pretrained_cfg
    return Wrapper().eval(), channels, tuple(cfg["mean"]), tuple(cfg["std"])


def export(name: str, out_dir: Path) -> Path:
    import onnxruntime as ort
    import torch

    model_id, kind, stride = CANDIDATES[name]
    out_grid = (GRID[0] + 2 * PAD, GRID[1] + 2 * PAD)
    input_size = (out_grid[0] * stride, out_grid[1] * stride)
    module, channels, mean, std = build(model_id, kind, stride, input_size)
    dummy = torch.randn(1, 3, input_size[1], input_size[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.onnx"
    kwargs = dict(input_names=["image"], output_names=["features"], opset_version=17)
    try:
        torch.onnx.export(module, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(module, dummy, str(path), **kwargs)
    BackboneSpec(name, input_size, stride, channels, mean, std, PAD).save(path.with_suffix(".json"))

    # The exported graph must reproduce PyTorch's output.
    with torch.no_grad():
        ref = module(dummy).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": dummy.numpy()})[0]
    if ref.shape != got.shape or ref.shape[2:] != (out_grid[1], out_grid[0]):
        raise RuntimeError(f"{name}: unexpected output shape {got.shape}, torch gave {ref.shape}")
    err = float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9))
    if err > 1e-3:
        raise RuntimeError(f"{name}: ONNX output differs from PyTorch (relative error {err:.2e})")
    size_mb = path.stat().st_size / 1e6
    print(f"{name:20s} {model_id:36s} input {input_size[0]}x{input_size[1]} -> "
          f"{out_grid[0]}x{out_grid[1]}x{channels} (cropped to {GRID[0]}x{GRID[1]})  "
          f"{size_mb:5.1f} MB  (onnx vs torch rel. err {err:.1e})")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of {', '.join(CANDIDATES)} (default: all)")
    ap.add_argument("--out", default="models")
    args = ap.parse_args(argv)
    unknown = set(args.names) - set(CANDIDATES)
    if unknown:
        ap.error(f"unknown model(s): {', '.join(sorted(unknown))}")
    for name in args.names or CANDIDATES:
        export(name, Path(args.out))


if __name__ == "__main__":
    main()
