"""Stereo calibration tool: capture checkerboard images, then calibrate.

Capture (on the drone computer; saves pairs automatically as you move the board):

    python -m tools.calibrate_stereo capture --left 0 --right 1 --out calib_captures

    --left/--right take a camera index, a device path, or a GStreamer pipeline
    ending in appsink. Add --preview if a display is attached.

Calibrate (anywhere; reads the captures):

    python -m tools.calibrate_stereo calibrate --captures calib_captures \
        --square 0.025 --intrinsics config/intrinsics.json --out config/stereo.json

Captures folder layout: pair_###_left.png / pair_###_right.png (board seen by
both cameras) and mono_###_left.png / mono_###_right.png (board seen by one
camera only). Mono captures matter for the wide camera: the stereo pairs only
cover the narrow camera's view, so put the board in the wide camera's corners
too, or its distortion is never measured there.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from perception.calibration import (BoardSpec, CameraModel, Rectifier, StereoCalibration,
                                    calibrate_camera, check_model, find_board, model_is_usable,
                                    per_pair_errors, stereo_calibrate)

DISPARITY_NOISE_PX = 0.2  # typical StereoBM sub-pixel noise, used for the depth accuracy estimate


def load_intrinsics(path: str | Path) -> dict[str, CameraModel | None]:
    """Read {"left": {...}, "right": {...}} with fx, fy, cx, cy, width, height and
    optional dist ([k1, k2, p1, p2, k3] or null). Missing cameras come back as None."""
    data = json.loads(Path(path).read_text())
    out = {}
    for side in ("left", "right"):
        c = data.get(side)
        out[side] = None if c is None else CameraModel.from_values(
            c["fx"], c["fy"], c["cx"], c["cy"], (c["width"], c["height"]), c.get("dist"))
    return out


def run_calibration(pairs: list[tuple[np.ndarray, np.ndarray]],
                    mono_left: list[np.ndarray], mono_right: list[np.ndarray],
                    board: BoardSpec, guess_left: CameraModel | None, guess_right: CameraModel | None,
                    refine: str = "auto", width: int = 640, log=print):
    """Detect, calibrate and rectify. Returns (StereoCalibration, Rectifier, summary dict).

    refine: "auto" re-estimates a camera's intrinsics when it has no usable
    distortion model (missing, overfit, or under-constrained); "always" / "never".
    """
    if not pairs:
        raise ValueError("no capture pairs")
    sizes = {(p[0].shape[1], p[0].shape[0]) for p in pairs}, {(p[1].shape[1], p[1].shape[0]) for p in pairs}
    if len(sizes[0]) != 1 or len(sizes[1]) != 1:
        raise ValueError("captures of one camera have mixed image sizes")
    size_l, size_r = next(iter(sizes[0])), next(iter(sizes[1]))

    log(f"detecting the board in {len(pairs)} pairs and {len(mono_left) + len(mono_right)} single captures ...")
    left_c, right_c, kept = [], [], []
    for i, (l_img, r_img) in enumerate(pairs):
        cl, cr = find_board(l_img, board), find_board(r_img, board)
        if cl is None or cr is None:
            continue
        if _order_disagrees(cl, cr):
            cr = cr[::-1].copy()
        left_c.append(cl), right_c.append(cr), kept.append(i)
    mono_l = [c for c in (find_board(img, board) for img in mono_left) if c is not None]
    mono_r = [c for c in (find_board(img, board) for img in mono_right) if c is not None]
    log(f"  board found in both images of {len(kept)}/{len(pairs)} pairs; "
        f"single captures used: left {len(mono_l)}, right {len(mono_r)}")
    if len(kept) < 8:
        raise ValueError(f"only {len(kept)} usable pairs; capture at least 15-20")

    models = {}
    for side, guess, size, stereo_c, mono_c in (("left", guess_left, size_l, left_c, mono_l),
                                                ("right", guess_right, size_r, right_c, mono_r)):
        if guess is not None and guess.size != size:
            if abs(guess.size[0] / guess.size[1] - size[0] / size[1]) > 1e-3:
                raise ValueError(f"{side} intrinsics are for {guess.size}, captures are {size}")
            log(f"  {side}: scaling intrinsics from {guess.size} to the capture size {size}")
            guess = guess.scaled(size)
        issues = check_model(guess) if guess is not None else ["no intrinsics given"]
        no_dist = guess is not None and not guess.dist.any()
        need = refine == "always" or (refine == "auto" and (guess is None or no_dist or issues))
        if guess is not None:
            for msg in issues:
                log(f"  {side} intrinsics: {msg}")
        if need:
            if guess is None:
                k = cv2.initCameraMatrix2D([board.object_points()] * len(stereo_c), stereo_c, size)
                guess = CameraModel(k, np.zeros(5), size)
            elif not model_is_usable(guess) or max(abs(guess.dist[2:4])) > 0.005:
                guess = CameraModel(guess.K.copy(), np.zeros(5), size)  # bad distortion is a bad start
            model, rms = calibrate_camera(board, stereo_c + mono_c, guess)
            log(f"  {side}: re-estimated intrinsics from {len(stereo_c) + len(mono_c)} views, rms {rms:.3f} px")
            models[side] = model
        else:
            models[side] = guess

    calib = stereo_calibrate(board, left_c, right_c, models["left"], models["right"])
    errs = per_pair_errors(calib, board, left_c, right_c)
    limit = max(1.0, 3 * float(np.median(errs)))
    bad = errs > limit
    if bad.any() and (~bad).sum() >= 8:
        log(f"  dropping {bad.sum()} pair(s) with error > {limit:.2f} px: "
            + ", ".join(str(kept[i]) for i in np.flatnonzero(bad)))
        left_c = [c for c, b in zip(left_c, bad) if not b]
        right_c = [c for c, b in zip(right_c, bad) if not b]
        calib = stereo_calibrate(board, left_c, right_c, models["left"], models["right"])

    rect = Rectifier(calib, (width, None))
    pl, pr = rect.rectify_points(np.concatenate(left_c), np.concatenate(right_c))
    row_err = np.abs(pl[:, 1] - pr[:, 1])
    summary = {
        "pairs_found": len(kept), "pairs_used": len(left_c), "rms_px": calib.rms,
        "baseline_m": rect.baseline, "rotation_deg": calib.rotation_deg, "swapped": rect.swapped,
        "rectified_size": list(rect.out_size), "rectified_f_px": rect.K[0, 0],
        "rectified_fov_deg": [math.degrees(2 * math.atan(rect.out_size[0] / 2 / rect.K[0, 0])),
                              math.degrees(2 * math.atan(rect.out_size[1] / 2 / rect.K[0, 0]))],
        "row_alignment_px": {"mean": float(row_err.mean()), "max": float(row_err.max())},
        "board": {"cols": board.cols, "rows": board.rows, "square_m": board.square},
    }
    calib.meta = summary
    return calib, rect, summary


def report(calib: StereoCalibration, rect: Rectifier, s: dict, log=print) -> None:
    log("\n=== stereo calibration ===")
    log(f"pairs used          {s['pairs_used']} of {s['pairs_found']} with the board in both")
    log(f"reprojection RMS    {s['rms_px']:.3f} px      (good: < 0.5)")
    log(f"baseline            {1000 * s['baseline_m']:.1f} mm   (check against a ruler)")
    log(f"camera misalignment {s['rotation_deg']:.2f} deg")
    log(f"row alignment       mean {s['row_alignment_px']['mean']:.3f} px, max {s['row_alignment_px']['max']:.3f} px "
        f"(good: mean < 0.2)")
    w, h = s["rectified_size"]
    log(f"rectified output    {w}x{h}, f = {s['rectified_f_px']:.1f} px, "
        f"view {s['rectified_fov_deg'][0]:.1f} x {s['rectified_fov_deg'][1]:.1f} deg")
    if s["swapped"]:
        log("NOTE: camera 2 is physically on the LEFT of camera 1; outputs are swapped so the "
            "pair is still left-to-right. Swapping the cables/indices would make this go away.")
    for side, m in (("left", calib.left), ("right", calib.right)):
        log(f"{side:6s} fx {m.K[0, 0]:.1f} fy {m.K[1, 1]:.1f} cx {m.K[0, 2]:.1f} cy {m.K[1, 2]:.1f}  "
            f"dist [{', '.join(f'{v:+.4f}' for v in m.dist.ravel())}]  view {m.fov_deg[0]:.0f} x {m.fov_deg[1]:.0f} deg")
        for msg in check_model(m):
            log(f"       warning: {msg}")
    log(f"expected per-pixel depth noise (1 sigma, assuming {DISPARITY_NOISE_PX} px disparity noise;"
        " ground plane and cell averages are far more precise):")
    fb = s["rectified_f_px"] * s["baseline_m"]
    for z in (2, 5, 10, 15, 20):
        log(f"   {z:>3d} m altitude: +/- {z * z * DISPARITY_NOISE_PX / fb * 100:.0f} cm")


def _order_disagrees(cl: np.ndarray, cr: np.ndarray) -> bool:
    """True if the two detections number the corners from opposite ends of the board."""
    vl, vr = cl[-1, 0] - cl[0, 0], cr[-1, 0] - cr[0, 0]
    return float(vl @ vr) < 0


def preview_image(rect: Rectifier, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Rectified pair side by side with horizontal lines: features should sit on the same line."""
    rl, rr = rect.rectify(left, right)
    img = cv2.cvtColor(np.hstack([rl, rr]), cv2.COLOR_GRAY2BGR)
    for y in range(16, img.shape[0], 32):
        cv2.line(img, (0, y), (img.shape[1] - 1, y), (0, 200, 0), 1)
    return img


def open_camera(spec: str, width: int, height: int) -> cv2.VideoCapture:
    if spec.isdigit():
        cap = cv2.VideoCapture(int(spec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    elif "!" in spec:
        cap = cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        sys.exit(f"could not open camera {spec!r}")
    return cap


def quick_find(img: np.ndarray, board: BoardSpec, scale: float = 0.5) -> np.ndarray | None:
    """Fast, coarse board check on a downscaled image (for deciding what to save)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, c = cv2.findChessboardCorners(small, (board.cols, board.rows), flags=cv2.CALIB_CB_FAST_CHECK)
    return c / scale if ok else None


def capture(args) -> None:
    board = BoardSpec(*_parse_board(args.board))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam_l, cam_r = (open_camera(s, args.width, args.height) for s in (args.left, args.right))
    n_pair = len(list(out.glob("pair_*_left.png")))
    n_mono = len(list(out.glob("mono_*_*.png")))
    last = {"pair": None, "left": None, "right": None}
    last_t = {"pair": 0.0, "left": 0.0, "right": 0.0}
    print(f"saving to {out}/ - move the board slowly; Ctrl+C or q to stop")
    try:
        while n_pair < args.count:
            # Grab both before decoding either, so the two exposures are as close as possible.
            if not (cam_l.grab() and cam_r.grab()):
                sys.exit("camera stopped delivering frames")
            l_img, r_img = cam_l.retrieve()[1], cam_r.retrieve()[1]
            cl, cr = quick_find(l_img, board), quick_find(r_img, board)
            now, saved = time.monotonic(), None
            if cl is not None and cr is not None:
                if _should_save(cl, last["pair"], last_t["pair"], now, l_img.shape[1], args.interval):
                    cv2.imwrite(str(out / f"pair_{n_pair:03d}_left.png"), l_img)
                    cv2.imwrite(str(out / f"pair_{n_pair:03d}_right.png"), r_img)
                    last["pair"], last_t["pair"], saved = cl, now, f"pair {n_pair}"
                    n_pair += 1
            elif args.mono:
                for side, c, img in (("left", cl, l_img), ("right", cr, r_img)):
                    if c is not None and _should_save(c, last[side], last_t[side], now, img.shape[1], args.interval):
                        cv2.imwrite(str(out / f"mono_{n_mono:03d}_{side}.png"), img)
                        last[side], last_t[side], saved = c, now, f"{side}-only {n_mono}"
                        n_mono += 1
            if saved:
                print(f"saved {saved}  (pairs {n_pair}/{args.count}, single {n_mono})")
            if args.preview:
                view = np.hstack([_draw(l_img, cl, board), _draw(r_img, cr, board)])
                cv2.imshow("calibration capture (q to quit)", cv2.resize(view, None, fx=0.4, fy=0.4))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    print(f"done: {n_pair} pairs, {n_mono} single-camera captures in {out}/")


def _should_save(c, last_c, last_t, now, width, interval) -> bool:
    if now - last_t < interval:
        return False
    # Require a real move so captures aren't near-duplicates.
    return last_c is None or float(np.linalg.norm(c - last_c, axis=-1).mean()) > 0.05 * width


def _draw(img, corners, board):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        cv2.drawChessboardCorners(out, (board.cols, board.rows), corners.astype(np.float32), True)
    return out


def calibrate(args) -> None:
    board = BoardSpec(*_parse_board(args.board), square=args.square)
    folder = Path(args.captures)
    pair_files = sorted(folder.glob("pair_*_left.png"))
    pairs = [(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE),
              cv2.imread(str(p).replace("_left.png", "_right.png"), cv2.IMREAD_GRAYSCALE)) for p in pair_files]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    mono_l = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(folder.glob("mono_*_left.png"))]
    mono_r = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(folder.glob("mono_*_right.png"))]
    guess = load_intrinsics(args.intrinsics) if args.intrinsics else {"left": None, "right": None}

    calib, rect, summary = run_calibration(pairs, mono_l, mono_r, board, guess["left"], guess["right"],
                                           refine=args.refine, width=args.width)
    report(calib, rect, summary)
    calib.save(args.out)
    print(f"\nsaved {args.out}")
    if pairs:
        prev = Path(args.out).with_name(Path(args.out).stem + "_rectified_check.png")
        cv2.imwrite(str(prev), preview_image(rect, *pairs[0]))
        print(f"saved {prev} (matching features should sit on the same green line)")


def _parse_board(text: str) -> tuple[int, int]:
    cols, rows = (int(v) for v in text.lower().split("x"))
    return cols, rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="save checkerboard captures from both cameras")
    c.add_argument("--left", required=True, help="camera index, device path or GStreamer pipeline")
    c.add_argument("--right", required=True)
    c.add_argument("--out", default="calib_captures")
    c.add_argument("--board", default="9x6", help="inner corners, cols x rows (default 9x6)")
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    c.add_argument("--count", type=int, default=30, help="stop after this many pairs")
    c.add_argument("--interval", type=float, default=1.0, help="min seconds between saves")
    c.add_argument("--no-mono", dest="mono", action="store_false",
                   help="don't save captures where only one camera sees the board")
    c.add_argument("--preview", action="store_true", help="show a live window (needs a display)")
    c.set_defaults(func=capture)

    k = sub.add_parser("calibrate", help="calibrate from a captures folder")
    k.add_argument("--captures", default="calib_captures")
    k.add_argument("--square", type=float, required=True, help="measured square size in metres")
    k.add_argument("--board", default="9x6")
    k.add_argument("--intrinsics", help="JSON with known intrinsics (see config/intrinsics.json)")
    k.add_argument("--refine", choices=["auto", "always", "never"], default="auto",
                   help="re-estimate intrinsics: auto = only when the given ones look unreliable")
    k.add_argument("--width", type=int, default=640, help="rectified output width")
    k.add_argument("--out", default="config/stereo.json")
    k.set_defaults(func=calibrate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
