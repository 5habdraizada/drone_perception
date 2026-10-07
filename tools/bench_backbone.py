"""Compare exported backbones: speed, a separability score, and a feature visualisation.

    python -m tools.bench_backbone                         # every models/*.onnx, synthetic scene
    python -m tools.bench_backbone --images frames/        # your own (rectified) frames
    python -m tools.bench_backbone models/resnet18.onnx --threads 4 --runs 50

Run it on the RB3 for the timings that matter.

The separability score uses the synthetic scene's ground truth (ground vs
structure vs pit). Its textures are procedural noise, so treat the score as a
sanity check only; the real test is the feature visualisation on real frames:
each cell's features are compressed to three colours (PCA), so cells the
network considers similar get similar colours. Good features make roofs,
roads, grass, water, etc. come out as clean, consistent colour regions.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from perception.backbone import Backbone
from sim.scenes import FRAME, RIG
from sim.synthetic_stereo import default_scene, render_stereo

CELL = 32
LABELS = ("ground", "structure", "pit")


def synthetic_frame() -> tuple[np.ndarray, np.ndarray]:
    """Left image of the default scene at the rectified size, plus per-cell labels (-1 = mixed)."""
    left, _, gt = render_stereo(default_scene(), RIG, size=FRAME, altitude=5.0, roll_deg=3, pitch_deg=-2)
    gh, gw = FRAME[1] // CELL, FRAME[0] // CELL
    h = gt["height"][: gh * CELL, : gw * CELL].reshape(gh, CELL, gw, CELL).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    med, spread = np.median(h, axis=-1), np.ptp(h, axis=-1)
    labels = np.select([med > 0.3, med < -0.3], [1, 2], 0)
    labels[spread > 0.3] = -1
    return left, labels


def separability(feats: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(leave-one-out nearest-centroid accuracy, between/within class cosine-distance ratio)."""
    keep = labels.ravel() >= 0
    x = feats.reshape(-1, feats.shape[-1])[keep]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    y = labels.ravel()[keep]
    classes = np.unique(y)
    sums = {c: x[y == c].sum(axis=0) for c in classes}
    counts = {c: (y == c).sum() for c in classes}
    correct = 0
    for i in range(len(y)):
        best, best_sim = None, -np.inf
        for c in classes:
            s, n = sums[c] - (x[i] if y[i] == c else 0), counts[c] - (y[i] == c)
            if n == 0:
                continue
            cen = s / n
            sim = float(x[i] @ cen / (np.linalg.norm(cen) + 1e-9))
            if sim > best_sim:
                best, best_sim = c, sim
        correct += best == y[i]
    within = np.mean([np.mean(1 - x[y == c] @ x[y == c].T) for c in classes if counts[c] > 1])
    between = np.mean(1 - x @ x.T)
    return correct / len(y), float(between / max(within, 1e-9))


def pca_rgb(feats: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gh, gw, c = feats.shape
    x = feats.reshape(-1, c).astype(np.float64)
    x -= x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-9), 0, 1).reshape(gh, gw, 3)
    return cv2.resize((rgb * 255).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="default: models/*.onnx")
    ap.add_argument("--images", help="folder of frames to visualise (default: synthetic scene)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "npu"])
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.models] or sorted(Path("models").glob("*.onnx"))
    if not paths:
        ap.error("no models found; run python -m tools.export_backbone first")

    syn_img, syn_labels = synthetic_frame()
    if args.images:
        files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        images = [cv2.imread(str(p)) for p in files[:6]]
    else:
        images = [syn_img]
    timing_img = cv2.resize(images[0], FRAME) if images[0].shape[:2] != FRAME[::-1] else images[0]

    print(f"{'model':20s} {'grid':>10s} {'ms/frame':>9s} {'p90':>6s} {'MB':>6s} {'sep. acc':>9s} {'ratio':>6s}")
    for path in paths:
        bb = Backbone(path, backend=args.backend, threads=args.threads)
        for _ in range(3):
            bb(timing_img)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            grid = bb(timing_img)
            times.append((time.perf_counter() - t0) * 1000)
        acc, ratio = separability(bb(syn_img).resample(syn_labels.shape), syn_labels)
        gh, gw = grid.shape
        print(f"{path.stem:20s} {f'{gw}x{gh}x{grid.features.shape[2]}':>10s} {np.median(times):9.1f} "
              f"{np.percentile(times, 90):6.1f} {path.stat().st_size / 1e6:6.1f} {acc:9.0%} {ratio:6.2f}")

        rows = []
        for img in images:
            shown = cv2.resize(img, FRAME) if img.ndim == 3 else cv2.cvtColor(cv2.resize(img, FRAME), cv2.COLOR_GRAY2BGR)
            rows.append(np.hstack([shown, pca_rgb(bb(img).features, FRAME)]))
        out = Path(args.out) / f"backbone_{path.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.vstack(rows))
    print(f"\nfeature visualisations saved to {args.out}/backbone_<model>.png")


if __name__ == "__main__":
    main()
