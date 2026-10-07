"""Run the perception pipeline on your own stereo data.

    python -m tools.run_data DATA [--calib config/stereo.json] [--model models/resnet18.onnx] [--out out/run]

DATA is a folder holding one of:
    left/ + right/              image folders (matched by file name, else by sorted order)
    left.mp4 + right.mp4        two videos (any of .mp4 .avi .mov .mkv)
    pair_000_left.png ...       the calibration tool's capture layout
and optionally:
    rig.json                    {"fx", "fy", "cx", "cy", "baseline"} for already-rectified images
    poses.csv                   frame,x,y,z,qw,qx,qy,qz per frame: the left camera's pose in the world
                                (z up, camera axes x right, y down, z forward); enables the map

Raw images need --calib (from tools/calibrate_stereo.py); rectified images need rig.json
or --rig. See the README section "Testing on your own data".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from perception.mapping import pose_from_quaternion
from perception.pipeline import Pipeline
from perception.terrain import StereoRig
from tools.viz import colourise, label, overlay, regions_bgr

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def stereo_frames(data: Path, left: str | None = None, right: str | None = None):
    """Yield (index, name, left_image, right_image) from any supported layout."""
    if left and right:
        return _pair_sources(Path(left), Path(right))
    if (data / "left").is_dir() and (data / "right").is_dir():
        return _pair_sources(data / "left", data / "right")
    videos = {p.stem: p for p in data.iterdir() if p.suffix.lower() in VIDEO_EXT}
    if "left" in videos and "right" in videos:
        return _pair_sources(videos["left"], videos["right"])
    lefts = sorted(p for p in data.glob("*_left.*") if p.suffix.lower() in IMAGE_EXT)
    if lefts:
        pairs = [(p, p.with_name(p.name.replace("_left.", "_right."))) for p in lefts]
        return ((i, l.stem[:-5], _imread(l), _imread(r)) for i, (l, r) in enumerate(pairs) if r.exists())
    raise SystemExit(f"no stereo data found in {data} (see --help for the supported layouts)")


def _pair_sources(left: Path, right: Path):
    if left.is_dir():
        lf = sorted(p for p in left.iterdir() if p.suffix.lower() in IMAGE_EXT)
        rf = sorted(p for p in right.iterdir() if p.suffix.lower() in IMAGE_EXT)
        rmap = {p.stem: p for p in rf}
        if all(p.stem in rmap for p in lf):
            pairs = [(p, rmap[p.stem]) for p in lf]
        else:
            if len(lf) != len(rf):
                print(f"warning: {len(lf)} left and {len(rf)} right images; pairing by sorted order")
            pairs = list(zip(lf, rf))
        return ((i, l.stem, _imread(l), _imread(r)) for i, (l, r) in enumerate(pairs))
    return _video_pairs(left, right)


def _video_pairs(left: Path, right: Path):
    cl, cr = cv2.VideoCapture(str(left)), cv2.VideoCapture(str(right))
    i = 0
    while True:
        ok_l, l_img = cl.read()
        ok_r, r_img = cr.read()
        if not (ok_l and ok_r):
            return
        yield i, f"{i:06d}", l_img, r_img
        i += 1


def _imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {path}")
    return img


def load_poses(path: Path) -> dict[int, np.ndarray]:
    poses = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = {k.strip(): float(x) for k, x in row.items()}
            poses[int(v["frame"])] = pose_from_quaternion(v["x"], v["y"], v["z"], v["qw"], v["qx"], v["qy"], v["qz"])
    return poses


def load_rig(path: Path) -> StereoRig:
    d = json.loads(Path(path).read_text())
    return StereoRig(d["fx"], d["fy"], d["cx"], d["cy"], d["baseline"])


def panel(res, height_range) -> np.ndarray:
    t = res.terrain
    tiles = [label(res.left, f"t = {res.t:.1f} s"),
             label(colourise(t.height, *height_range), f"height {height_range[0]:g}..{height_range[1]:g} m")]
    if res.scene is not None:
        tiles += [label(overlay(res.left, regions_bgr(res.scene.regions)), f"regions ({len(res.scene.region_info)})"),
                  label(colourise(res.scene.novelty, 0.0, 6.0, cv2.COLORMAP_INFERNO), "novelty z 0..6")]
    return np.hstack(tiles)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="folder with the stereo data")
    ap.add_argument("--calib", help="stereo calibration JSON (raw images)")
    ap.add_argument("--rig", help="rig JSON for already-rectified images (default: DATA/rig.json)")
    ap.add_argument("--left", help="left image folder or video (overrides the layout detection)")
    ap.add_argument("--right", help="right image folder or video")
    ap.add_argument("--poses", help="poses CSV (default: DATA/poses.csv if present)")
    ap.add_argument("--model", help="backbone .onnx for regions and novelty (default: terrain and heights only)")
    ap.add_argument("--width", type=int, default=640, help="rectified width when using --calib")
    ap.add_argument("--fps", type=float, default=10.0, help="frame rate, for timestamps")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--save-every", type=int, default=10, help="save a panel image every n processed frames")
    ap.add_argument("--height-range", type=float, nargs=2, default=(-1.0, 3.0))
    ap.add_argument("--out", default="out/run")
    args = ap.parse_args(argv)

    data, out = Path(args.data), Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = stereo_frames(data, args.left, args.right)

    poses_path = Path(args.poses) if args.poses else data / "poses.csv"
    poses = load_poses(poses_path) if poses_path.exists() else {}
    first = None
    pipe = None
    rows, done = [], 0
    for i, name, l_img, r_img in frames:
        if i < args.start or (i - args.start) % args.step:
            continue
        if pipe is None:
            pipe = _make_pipeline(args, data, l_img)
            first = i
        res = pipe.process(i / args.fps, l_img, r_img, poses.get(i))
        t = res.terrain
        nov = res.scene.novelty if res.scene is not None else None
        rows.append({"frame": i, "name": name, "t": round(res.t, 3),
                     "altitude_m": _r(t.altitude), "tilt_deg": _r(t.tilt_deg),
                     "depth_coverage": _r(t.valid_fraction), "ground_inliers": _r(t.inlier_fraction),
                     "regions": len(res.scene.region_info) if res.scene is not None else "",
                     "novelty_max": _r(np.nanmax(nov)) if nov is not None and np.isfinite(nov).any() else "",
                     "mapped": int(res.mapped), **{f"{k}_ms": _r(v, 1) for k, v in res.timings_ms.items()}})
        if done % args.save_every == 0:
            cv2.imwrite(str(out / "frames" / f"frame_{i:06d}.png"), panel(res, args.height_range))
        done += 1
        print(f"\rframe {i}: altitude {t.altitude:5.2f} m, coverage {t.valid_fraction:4.0%}", end="", flush=True)
        if args.max_frames and done >= args.max_frames:
            break
    print()
    if not rows:
        raise SystemExit("no frames processed")

    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = _summary(rows, pipe, first)
    if pipe.map.frames:
        cv2.imwrite(str(out / "map.png"), pipe.map.render(args.height_range, scale=4))
        (out / "anomalies.json").write_text(json.dumps(pipe.map.anomalies(), indent=2))
    (out / "summary.txt").write_text(summary)
    print(summary)
    print(f"results in {out}/")


def _make_pipeline(args, data: Path, l_img: np.ndarray) -> Pipeline:
    if args.calib:
        return Pipeline.from_calibration(args.calib, width=args.width, backbone_path=args.model)
    rig_path = Path(args.rig) if args.rig else data / "rig.json"
    if not rig_path.exists():
        raise SystemExit("raw images need --calib; already-rectified images need --rig or DATA/rig.json")
    return Pipeline(load_rig(rig_path), (l_img.shape[1], l_img.shape[0]), backbone_path=args.model)


def _summary(rows, pipe: Pipeline, first: int) -> str:
    col = lambda k: np.array([r[k] for r in rows if r[k] != "" and r[k] is not None], float)
    alt, cov = col("altitude_m"), col("depth_coverage")
    lines = [f"frames processed      {len(rows)} (from frame {first})",
             f"ground plane found    {np.isfinite(alt).sum()} of {len(rows)} frames",
             f"altitude              median {np.nanmedian(alt):.2f} m, range {np.nanmin(alt):.2f}..{np.nanmax(alt):.2f} m"
             if np.isfinite(alt).any() else "altitude              n/a",
             f"depth coverage        median {np.median(cov):.0%}"]
    for k in ("rectify_ms", "terrain_ms", "scene_ms", "map_ms"):
        lines.append(f"{k[:-3] + ' time':22s}median {np.median(col(k)):.1f} ms")
    if pipe.map.frames:
        found = pipe.map.anomalies()
        lines.append(f"mapped frames         {pipe.map.frames}; confirmed anomalies {len(found)}")
        lines += [f"  ({a['x']:+.1f}, {a['y']:+.1f}) m  peak z {a['peak_novelty']:.1f}  hits {a['hits']}  "
                  f"height {a['height']:.2f} m" for a in found]
    else:
        lines.append("map                   not built (no poses.csv)")
    return "\n".join(lines)


def _r(v, nd=3):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


if __name__ == "__main__":
    main()
