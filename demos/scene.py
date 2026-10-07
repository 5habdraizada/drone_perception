"""Regions and novelty along a synthetic flight (real terrain + backbone + scene model).

    python -m demos.scene [--model models/resnet18.onnx]

Expect stable region colours and a novelty spike on the striped object when it
comes into view. Steep roof walls also score high early on: this renderer draws
them as stretched texture unlike anything seen before them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from perception.pipeline import Pipeline
from sim.scenes import FRAME, RIG, flight_scene
from sim.synthetic_stereo import render_stereo
from tools.viz import colourise, label, overlay, regions_bgr


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/resnet18.onnx")
    ap.add_argument("--altitude", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=1.0, help="metres flown between frames")
    ap.add_argument("--out", default="out/scene_demo.png")
    args = ap.parse_args(argv)

    scene, pipe = flight_scene(), Pipeline(RIG, FRAME, backbone_path=args.model)
    xs = np.arange(-9.0, 9.0 + 1e-6, args.step)
    show = {int(i) for i in np.linspace(0, len(xs) - 1, 6)}
    rows = []
    print(f"{'x (m)':>6s} {'regions':>7s} {'max novelty':>11s}   terrain  scene")
    for i, x in enumerate(xs):
        left, right, _ = render_stereo(scene, RIG, size=FRAME, altitude=args.altitude, roll_deg=1.5 * np.sin(i),
                                       pitch_deg=1.0 * np.cos(i), position=(x, 0.0))
        res = pipe.process(float(i), left, right)
        nov = res.scene.novelty
        peak = f"{np.nanmax(nov):11.1f}" if np.isfinite(nov).any() else f"{'warm-up':>11s}"
        print(f"{x:6.1f} {len(res.scene.region_info):7d} {peak}   "
              f"{res.timings_ms['terrain']:5.0f} ms {res.timings_ms['scene']:4.0f} ms")
        if i in show:
            rows.append(np.hstack([
                label(left, f"x = {x:+.1f} m"),
                label(overlay(left, regions_bgr(res.scene.regions)), f"regions ({len(res.scene.region_info)})"),
                label(colourise(res.terrain.cell_height, 0, 2.5, cv2.COLORMAP_INFERNO), "height (0..2.5 m)"),
                label(colourise(nov, 0, 8, cv2.COLORMAP_INFERNO), "novelty (z 0..8)")]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
