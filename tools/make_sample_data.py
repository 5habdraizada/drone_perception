"""Write a synthetic flight in the layout tools/run_data.py reads, with exact poses.

    python -m tools.make_sample_data [--out out/sample_data] [--frames 12] [--raw]

Default: rectified 640 × 352 pairs in left/ and right/, rig.json and poses.csv.
--raw: unrectified, distorted 960 × 540 images from a mismatched long/wide lens
pair like the real rig, plus the matching stereo.json for --calib.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from perception.calibration import CameraModel, StereoCalibration
from perception.mapping import quaternion_from_rotation
from sim.scenes import FRAME, RIG, flight_scene
from sim.synthetic_stereo import camera_pose, render_stereo, render_view

RAW = (960, 540)


def sample_calibration() -> StereoCalibration:
    """The project's camera matrices with plausible distortion, at half resolution."""
    left = CameraModel.from_values(1718.54, 1720.12, 1054.17, 454.72, (1920, 1080), [-0.06, 0.10, 0.0002, -0.0001, 0.0])
    right = CameraModel.from_values(1175.14, 1177.06, 919.18, 602.39, (1920, 1080), [-0.20, 0.10, 0.0005, -0.0005, -0.02])
    r = cv2.Rodrigues(np.radians([0.8, -0.5, 0.3]))[0]
    return StereoCalibration(left.scaled(RAW), right.scaled(RAW), r, -r @ np.array([0.10, 0.002, -0.001]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/sample_data")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--altitude", type=float, default=6.0)
    ap.add_argument("--raw", action="store_true", help="unrectified, distorted images plus stereo.json")
    args = ap.parse_args(argv)

    out = Path(args.out)
    for side in ("left", "right"):
        (out / side).mkdir(parents=True, exist_ok=True)
    scene = flight_scene()
    calib = sample_calibration() if args.raw else None
    if calib is not None:
        calib.save(out / "stereo.json")
    else:
        (out / "rig.json").write_text(json.dumps(
            {"fx": RIG.fx, "fy": RIG.fy, "cx": RIG.cx, "cy": RIG.cy, "baseline": RIG.baseline}, indent=2))

    rng = np.random.default_rng(0)
    with open(out / "poses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
        for i, x in enumerate(np.linspace(-8.0, 8.0, args.frames)):
            roll, pitch = 1.5 * np.sin(i), 1.0 * np.cos(i)
            r_wc, c = camera_pose(args.altitude, roll, pitch, (x, 0.0))
            if calib is None:
                left, right, _ = render_stereo(scene, RIG, size=FRAME, altitude=args.altitude,
                                               roll_deg=roll, pitch_deg=pitch, position=(x, 0.0), seed=i)
            else:
                left, _, _ = render_view(scene, calib.left.K, calib.left.dist, RAW, r_wc, c, rng=rng)
                right, _, _ = render_view(scene, calib.right.K, calib.right.dist, RAW, r_wc @ calib.R.T,
                                          c + r_wc @ (-calib.R.T @ calib.T), rng=rng)
            cv2.imwrite(str(out / "left" / f"{i:06d}.png"), left)
            cv2.imwrite(str(out / "right" / f"{i:06d}.png"), right)
            w.writerow([i, *np.round(c, 4), *np.round(quaternion_from_rotation(r_wc), 6)])
            print(f"\rframe {i + 1}/{args.frames}", end="", flush=True)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
