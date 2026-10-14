"""Convert the QUT Kagaru Airborne Stereo dataset subset into tools/run_data.py's layout.

    python -m tools.prepare_kagaru [--src data/kagaru] [--out data/kagaru/run]

Expects in --src (from https://michaelwarren.info/docs/datasets/kagaru-airborne-stereo/):
    100831_155323_MultiCamera0_subset/   raw Bayer BMPs (cam0 = front, cam1 = rear)
    100831_155323_MultiCamera0.log       image timestamps
    100831_155323_Ins0.log               XSens INS/GPS
    cam0/1_intrinsics.xml, cam0/1_distortion.xml

Writes left/ and right/ (demosaiced greyscale), stereo.json and ins.csv. The published
extrinsics are ~10 px off in flight (vibration, per the dataset page), so the cameras'
relative pose is re-estimated from the footage with refine_extrinsics, keeping the
published 776 mm baseline.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from perception.calibration import CameraModel, StereoCalibration, check_model, refine_extrinsics

BASELINE_M = 0.776      # |t| of the published stereo extrinsics (6.09, -775.97, 7.37) mm
SIZE = (1280, 960)
BAYER = cv2.COLOR_BayerGR2GRAY  # the smooth-green-channel pattern for this BayerGR8 sensor


def read_matrix(path: Path) -> np.ndarray:
    data = re.search(r"<data>(.*?)</data>", path.read_text(), re.S).group(1)
    return np.array(data.split(), float)


def frame_times(log: Path) -> dict[int, float]:
    times = {}
    for line in log.read_text().splitlines()[1:]:
        p = line.split()
        times[int(re.search(r"image(\d+)", p[10]).group(1))] = int(p[0]) + int(p[1]) * 1e-6
    return times


def ground_elevation(ins: np.ndarray) -> float:
    """Altitude AMSL while parked on the airstrip: GPS fixed, near-zero speed, before take-off."""
    fixed = np.abs(ins[:, 2]) > 1.0                       # latitude is ~0 until the GPS has a fix
    speed = np.linalg.norm(ins[:, 6:9], axis=1)
    first_fast = np.argmax(fixed & (speed > 5.0))
    parked = fixed & (speed < 1.0) & (np.arange(len(ins)) < first_fast)
    if parked.sum() < 20:
        raise SystemExit("couldn't find the parked period before take-off in the INS log")
    return float(np.median(ins[parked, 4]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/kagaru")
    ap.add_argument("--out", default="data/kagaru/run")
    ap.add_argument("--refine-frames", type=int, default=10, help="frames used to re-estimate the extrinsics")
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    sub = next(src.glob("*_subset"))

    models = [CameraModel(read_matrix(src / f"cam{i}_intrinsics.xml").reshape(3, 3),
                          read_matrix(src / f"cam{i}_distortion.xml"), SIZE) for i in (0, 1)]
    for i, m in enumerate(models):
        for msg in check_model(m):
            print(f"cam{i}: {msg}")

    pairs = {}
    for p in sorted(sub.glob("cam0_image*.bmp")):
        k = int(re.search(r"(\d+)", p.stem.split("image")[1]).group(1))
        q = sub / p.name.replace("cam0_", "cam1_")
        if q.exists():
            pairs[k] = (p, q)
    (out / "left").mkdir(parents=True, exist_ok=True)
    (out / "right").mkdir(parents=True, exist_ok=True)
    grey = {}
    for k, (p, q) in pairs.items():
        grey[k] = tuple(cv2.cvtColor(cv2.imread(str(x), cv2.IMREAD_GRAYSCALE), BAYER) for x in (p, q))
        cv2.imwrite(str(out / "left" / f"{k:05d}.png"), grey[k][0])
        cv2.imwrite(str(out / "right" / f"{k:05d}.png"), grey[k][1])
    print(f"{len(pairs)} stereo pairs demosaiced into {out}")

    keys = sorted(grey)
    pick = [keys[i] for i in np.linspace(0, len(keys) - 1, min(args.refine_frames, len(keys))).astype(int)]
    # No usable prior pose: estimate it from scratch, then polish the rotation with that direction fixed.
    guess = StereoCalibration(models[0], models[1], np.eye(3), np.array([BASELINE_M, 0.0, 0.0]))
    calib = refine_extrinsics(guess, [grey[k] for k in pick], mode="full")
    calib = refine_extrinsics(calib, [grey[k] for k in pick], mode="rotation")
    calib.meta["source"] = "QUT Kagaru Airborne Stereo, intrinsics as published, extrinsics refined from footage"
    calib.save(out / "stereo.json")

    ins = np.loadtxt(src / "100831_155323_Ins0.log")
    t_ins = ins[:, 0] + ins[:, 1] * 1e-6
    ground = ground_elevation(ins)
    times = frame_times(src / "100831_155323_MultiCamera0.log")
    with open(out / "ins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "lat_deg", "lon_deg", "alt_amsl_m", "agl_m", "v_east", "v_north", "v_up",
                    "roll_rad", "pitch_rad", "yaw_rad"])
        for idx, k in enumerate(keys):
            t = times[k]
            i = int(np.clip(np.searchsorted(t_ins, t), 1, len(t_ins) - 1))
            j = i - 1 if abs(t_ins[i - 1] - t) < abs(t_ins[i] - t) else i
            r = ins[j]
            w.writerow([idx, f"{t:.6f}", r[2], r[3], r[4], round(r[4] - ground, 3), *r[6:12]])
    print(f"ground elevation {ground:.1f} m AMSL (parked before take-off); wrote ins.csv")


if __name__ == "__main__":
    main()
