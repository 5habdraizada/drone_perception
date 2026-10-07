"""Terrain on one synthetic stereo pair, compared with ground truth.

    python -m demos.terrain [--altitude 5] [--roll 4] [--pitch -3] [--width 640] [--matcher bm|sgbm]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from perception.terrain import StereoRig, TerrainConfig, TerrainEstimator
from sim.synthetic_stereo import default_scene, render_stereo
from tools.viz import colourise, label


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--altitude", type=float, default=5.0)
    ap.add_argument("--roll", type=float, default=4.0)
    ap.add_argument("--pitch", type=float, default=-3.0)
    ap.add_argument("--width", type=int, default=640, help="image width; height is 3/4 of it")
    ap.add_argument("--matcher", default="bm", choices=["bm", "sgbm"])
    ap.add_argument("--out", default="out/terrain_demo.png")
    args = ap.parse_args(argv)

    w, h = args.width, args.width * 3 // 4
    f = 300.0 * w / 320  # ~56 deg field of view at any resolution
    rig = StereoRig(fx=f, fy=f, cx=w / 2, cy=h / 2, baseline=0.10)
    left, right, gt = render_stereo(default_scene(), rig, size=(w, h), altitude=args.altitude,
                                    roll_deg=args.roll, pitch_deg=args.pitch)
    est = TerrainEstimator(rig, TerrainConfig.for_width(w, matcher=args.matcher))
    t = est.process(left, right)
    t0 = time.perf_counter()
    for _ in range(20):
        est.process(left, right)
    ms = (time.perf_counter() - t0) / 20 * 1000

    err = np.abs(t.height - gt["height"])
    print(f"altitude {t.altitude:.3f} m (true {args.altitude:.3f}), tilt {t.tilt_deg:.2f} deg")
    print(f"depth coverage {t.valid_fraction:.0%}, ground inliers {t.inlier_fraction:.0%}")
    print(f"height error: median {np.nanmedian(err):.3f} m, 90th percentile {np.nanpercentile(err, 90):.3f} m")
    print(f"{ms:.1f} ms per frame on this machine")

    size = (320, 240)
    rows = [[label(left, f"left image ({w}x{h})", size),
             label(colourise(t.disparity, np.nanmin(t.disparity), np.nanmax(t.disparity)), "disparity (px)", size),
             label(colourise(gt["height"], -1.0, 1.6), "height: ground truth", size)],
            [label(colourise(t.height, -1.0, 1.6), "height: estimated (-1..1.6 m)", size),
             label(colourise(t.slope_deg, 0, 60, cv2.COLORMAP_INFERNO), "slope (0..60 deg)", size),
             label(colourise(err, 0, 0.3, cv2.COLORMAP_INFERNO), "height error (0..0.3 m)", size)]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(r) for r in rows]))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
