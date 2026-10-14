"""Fly the pipeline over a real place (orthophoto + LiDAR from tools/fetch_pdok.py) and score it.

    python -m demos.real_area [--area data/delft] [--altitude 45] [--baselines 0.10 0.30 0.50]

Renders a straight flight line with the project's camera (640 x 352, f = 577 px) for each
stereo baseline, runs terrain + map with exact poses, and compares altitude and the mapped
heights with the LiDAR truth: how well does each baseline see buildings from this height?
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np

from perception.mapping import MapConfig
from perception.pipeline import Pipeline
from perception.terrain import TerrainConfig
from sim.real_world import load_area
from sim.scenes import FRAME, RIG
from sim.synthetic_stereo import camera_pose, render_view
from tools.viz import colourise


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="data/delft")
    ap.add_argument("--altitude", type=float, default=45.0, help="m above street level")
    ap.add_argument("--baselines", type=float, nargs="+", default=(0.10, 0.30, 0.50))
    ap.add_argument("--y", type=float, default=40.0, help="flight line's north offset from the centre, m")
    ap.add_argument("--x-range", type=float, nargs=2, default=(-120.0, 120.0))
    ap.add_argument("--spacing", type=float, default=8.0, help="m between frames")
    ap.add_argument("--out", default="out/real_area")
    ap.add_argument("--verbose", action="store_true", help="print each frame's altitude per baseline")
    args = ap.parse_args(argv)
    cache = Path(args.out) / "renders"
    cache.mkdir(parents=True, exist_ok=True)

    area = load_area(args.area)
    k = np.array([[RIG.fx, 0, RIG.cx], [0, RIG.fy, RIG.cy], [0, 0, 1.0]])
    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-6, args.spacing)
    rng = np.random.default_rng(0)
    tallest = float(area.truth_height.max())
    min_depth = max(2.0, args.altitude - tallest - 2.0)
    pipes = {b: Pipeline(dataclasses.replace(RIG, baseline=b), FRAME,
                         terrain_config=TerrainConfig.for_width(FRAME[0], min_depth=min_depth, max_depth=3 * args.altitude),
                         map_config=MapConfig(resolution=0.5)) for b in args.baselines}
    alt_err = {b: [] for b in args.baselines}

    print(f"{len(xs)} frames at {args.altitude:.0f} m (tallest building {tallest:.0f} m); baselines {list(args.baselines)} m")
    for i, x in enumerate(xs):
        roll, pitch = 1.0 * math.sin(i), 0.8 * math.cos(i)
        r_wc, c = camera_pose(args.altitude, roll, pitch, (x, args.y))
        # Renders are cached: they depend only on the area, the pose and the baseline.
        tag = f"{Path(args.area).name}_{args.altitude:g}_{x:g}_{args.y:g}"
        left = _render(cache / f"{tag}_left.png", area.scene, k, r_wc, c, rng)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r_wc, c
        line = []
        for b, pipe in pipes.items():
            right = _render(cache / f"{tag}_right{b:g}.png", area.scene, k, r_wc, c + r_wc @ np.array([b, 0.0, 0.0]), rng)
            res = pipe.process(float(i), left, right, pose)
            alt_err[b].append(res.terrain.altitude - args.altitude)
            line.append(f"{b:g} m: {res.terrain.altitude:5.1f}")
        if args.verbose:
            # Share of the camera's ground footprint covered by buildings (> 5 m) in the LiDAR truth.
            hx, hy = FRAME[0] / 2 / RIG.fx * args.altitude, FRAME[1] / 2 / RIG.fy * args.altitude
            gx, gy = np.meshgrid(np.linspace(x - hx, x + hx, 40), np.linspace(args.y - hy, args.y + hy, 22))
            roof = np.nanmean(area.height_at(gx, gy) > 5.0)
            print(f"x {x:+6.1f}  roof {roof:4.0%}  altitude " + ", ".join(line))
        else:
            print(f"\rframe {i + 1}/{len(xs)}", end="", flush=True)
    print()

    print(f"\n{'baseline':>8s} {'disparity':>10s} {'plane found':>11s} {'altitude err':>12s} "
          f"{'ground err':>10s} {'roof err':>9s} {'buildings seen':>14s} {'mapped':>7s}")
    rows = []
    for b, pipe in pipes.items():
        g = pipe.map
        rr, cc = np.nonzero(g.seen > 0)
        wx, wy = g.cell_to_world(rr, cc)
        truth = area.height_at(wx, wy)
        mapped = g.height[rr, cc]
        has = np.isfinite(mapped) & np.isfinite(truth)
        ground, roof = has & (truth < 0.5), has & (truth > 5.0)
        seen_bldg = (np.isfinite(truth) & (truth > 5.0))
        detected = (mapped[seen_bldg] > 2.5).mean() if seen_bldg.any() else math.nan
        e = np.array(alt_err[b], float)
        print(f"{b:8.2f} {RIG.fx * b / args.altitude:8.2f}px {np.isfinite(e).mean():11.0%} "
              f"{np.nanmedian(np.abs(e)):10.2f} m {np.median(np.abs(mapped[ground] - truth[ground])):8.2f} m "
              f"{np.median(np.abs(mapped[roof] - truth[roof])) if roof.any() else math.nan:7.2f} m "
              f"{detected:14.0%} {has.mean():7.0%}")
        rows.append((b, g))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = []
    for b, g in rows:
        rr, cc = np.nonzero(g.seen > 0)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        h = np.flipud(g.height[r0:r1, c0:c1])
        wx, wy = g.cell_to_world(*np.mgrid[r0:r1, c0:c1])
        t = np.flipud(area.height_at(wx, wy))
        scale = 3
        big = lambda a: cv2.resize(colourise(a, 0, 30), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if not panels:
            panels.append(_titled(big(np.where(np.isfinite(h), t, np.nan)), "LiDAR truth (0..30 m)"))
        panels.append(_titled(big(h), f"mapped, baseline {b:.2f} m"))
    width = max(p.shape[1] for p in panels)  # each baseline's map covers a slightly different area
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for p in panels]
    cv2.imwrite(str(out / "heights.png"), np.vstack(panels))
    print(f"\nsaved {out / 'heights.png'}")


def _render(path: Path, scene, k, r_wc, c, rng) -> np.ndarray:
    if path.exists():
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img, _, _ = render_view(scene, k, None, FRAME, r_wc, c, rng=rng)
    cv2.imwrite(str(path), img)
    return img


def _titled(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((22, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


if __name__ == "__main__":
    main()
