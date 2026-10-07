"""The full chain into the map along a synthetic flight, checked against ground truth.

    python -m demos.map [--model models/resnet18.onnx]

Poses come from the simulator, standing in for the control team.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from perception.pipeline import Pipeline
from sim.scenes import FRAME, RIG, flight_scene
from sim.synthetic_stereo import camera_pose, render_stereo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/resnet18.onnx")
    ap.add_argument("--altitude", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--out", default="out/map_demo.png")
    args = ap.parse_args(argv)

    scene, pipe = flight_scene(), Pipeline(RIG, FRAME, backbone_path=args.model)
    xs = np.arange(-9.0, 9.0 + 1e-6, args.step)
    total = 0.0
    for i, x in enumerate(xs):
        roll, pitch = 1.5 * np.sin(i), 1.0 * np.cos(i)
        left, right, _ = render_stereo(scene, RIG, size=FRAME, altitude=args.altitude, roll_deg=roll,
                                       pitch_deg=pitch, position=(x, 0.0))
        r_wc, c = camera_pose(args.altitude, roll, pitch, (x, 0.0))
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r_wc, c
        res = pipe.process(float(i), left, right, pose)
        total += sum(res.timings_ms.values())
    grid = pipe.map
    print(f"{len(xs)} frames, {total / len(xs):.0f} ms per frame on this machine")

    h = grid.height
    rr, cc = np.nonzero(np.isfinite(h))
    wx, wy = grid.cell_to_world(rr, cc)
    truth = scene.sample(wx.astype(np.float32), wy.astype(np.float32), scene.height, cv2.INTER_NEAREST).ravel()
    err = np.abs(h[rr, cc] - truth)
    print(f"height layer vs truth over {len(err)} cells: median {np.median(err):.3f} m, "
          f"90th percentile {np.percentile(err, 90):.3f} m")

    odd = next(b for b in scene.blocks if b.pattern == "stripes")
    found = grid.anomalies()
    print(f"\nconfirmed anomalies: {len(found)} (the striped object is at ({odd.x:+.1f}, {odd.y:+.1f}))")
    for a in found:
        tag = "  <- striped object" if np.hypot(a["x"] - odd.x, a["y"] - odd.y) < 1.0 else ""
        print(f"  ({a['x']:+5.1f}, {a['y']:+5.1f}) m  area {a['area_m2']:4.1f} m2  peak z {a['peak_novelty']:4.1f}  "
              f"hits {a['hits']}  height {a['height']:4.2f} m{tag}")

    panel = grid.render(scale=6)
    w = (panel.shape[1] - 3 * 4) // 4
    rows = []
    for k, name in enumerate(["height above ground (-1..2.5 m)", "regions", "anomaly hits", "times seen"]):
        title = np.zeros((22, w, 3), np.uint8)
        cv2.putText(title, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows += [title, panel[:, k * (w + 4): k * (w + 4) + w]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
