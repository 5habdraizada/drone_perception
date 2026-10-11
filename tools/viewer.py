"""Perception viewer: run the pipeline on a recording and watch every output in the browser.

    python -m tools.viewer DATA [--calib config/stereo.json] [--model models/resnet18.onnx]
    then open http://127.0.0.1:8000

Takes the same data and options as tools/run_data.py. Shows both rectified cameras,
disparity, height, slope, roughness, regions, novelty and contrast for the current
frame, the frame's measurements and timings, the regions and anomalies, charts over
time, and the world map when poses are given. Play, pause, step and scrub back
through processed frames.

--host 0.0.0.0 makes it reachable from other machines (e.g. viewing the drone computer
from a laptop). There is no authentication: only do that on a network you trust.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from tools.run_data import add_data_arguments, frame_record, make_pipeline, poses_for, selected_frames
from tools.viz import PALETTE, colourise, overlay, regions_bgr

HTML = Path(__file__).with_name("viewer.html")
LAYERS = ["left", "right", "disparity", "height", "slope", "roughness", "regions", "novelty", "contrast"]


class Session:
    """Runs the pipeline in a background thread and keeps recent frames' outputs for the UI."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()        # guards the state the UI reads
        self.step_lock = threading.Lock()   # one frame at a time; reset waits for it
        self.reset()
        threading.Thread(target=self._worker, daemon=True).start()

    def reset(self) -> None:
        with self.step_lock, self.lock:
            self.frames = selected_frames(self.args)
            self.poses = poses_for(self.args)
            self.pipe = None
            self.cache: OrderedDict[int, dict] = OrderedDict()   # frame -> {"layers", "record", "extra"}
            self.series: list[dict] = []
            self.path: list[tuple[float, float]] = []
            self.current, self.playing, self.done, self.error = None, False, False, None
            self.legend: dict[str, str] = {}

    def control(self, action: str, value=None) -> None:
        with self.lock:
            if action == "play":
                self.playing = not self.done
            elif action == "pause":
                self.playing = False
            elif action == "seek" and value is not None and int(value) in self.cache:
                self.playing, self.current = False, int(value)
            elif action in ("back", "forward") and self.current is not None:
                keys = list(self.cache)
                k = keys.index(self.current) + (-1 if action == "back" else 1)
                if 0 <= k < len(keys):
                    self.playing, self.current = False, keys[k]
                    return
        if action == "step" or (action == "forward" and self.current == self._last()):
            with self.lock:
                self.playing = False
            self._advance()
        elif action == "reset":
            self.reset()

    def _last(self):
        return next(reversed(self.cache)) if self.cache else None

    def _worker(self) -> None:
        while True:
            if self.playing and not self.done:
                t0 = time.perf_counter()
                self._advance()
                if self.args.play_fps:
                    time.sleep(max(0.0, 1.0 / self.args.play_fps - (time.perf_counter() - t0)))
            else:
                time.sleep(0.05)

    def _advance(self) -> None:
        with self.step_lock:
            self._advance_locked()

    def _advance_locked(self) -> None:
        try:
            i, name, l_img, r_img = next(self.frames)
        except StopIteration:
            with self.lock:
                self.done, self.playing = True, False
            return
        try:
            if self.pipe is None:
                self.pipe = make_pipeline(self.args, l_img)
            pose = self.poses.get(i)
            res = self.pipe.process(i / self.args.fps, l_img, r_img, pose)
            layers, legend = render_layers(res, self.args.height_range)
            record = frame_record(i, name, res)
            extra = frame_extra(res)
        except Exception as e:  # keep the UI alive and show what went wrong
            with self.lock:
                self.error, self.playing = f"frame {i}: {type(e).__name__}: {e}", False
            return
        encoded = {k: cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes() for k, v in layers.items()}
        with self.lock:
            self.cache[i] = {"layers": encoded, "record": record, "extra": extra}
            while len(self.cache) > self.args.keep:
                self.cache.popitem(last=False)
            self.series.append({k: record[k] for k in ("frame", "altitude_m", "depth_coverage", "novelty_max",
                                                       "ground_inliers")}
                               | {"total_ms": round(sum(res.timings_ms.values()), 1)})
            if pose is not None:
                self.path.append((float(pose[0, 3]), float(pose[1, 3])))
            self.legend, self.current, self.error = legend, i, None

    def state(self) -> dict:
        with self.lock:
            cur = self.cache.get(self.current) if self.current is not None else None
            grid = self.pipe.map if self.pipe is not None else None
            return {
                "data": str(self.args.data), "frame": self.current, "playing": self.playing, "done": self.done,
                "error": self.error, "cached": list(self.cache), "processed": len(self.series),
                "record": cur["record"] if cur else None, "extra": cur["extra"] if cur else None,
                "legend": self.legend, "series": self.series,
                "has_map": bool(grid is not None and grid.frames),
                "anomalies": _jsonable(grid.anomalies()) if grid is not None and grid.frames else [],
                "map_frames": grid.frames if grid is not None else 0,
            }

    def layer(self, frame: int, name: str) -> bytes | None:
        with self.lock:
            entry = self.cache.get(frame)
            return entry["layers"].get(name) if entry else None

    def map_image(self, layer: str) -> bytes | None:
        with self.lock:
            if self.pipe is None or not self.pipe.map.frames:
                return None
            img = render_map(self.pipe.map, layer, self.path, self.args.height_range)
        return cv2.imencode(".png", img)[1].tobytes()


def render_layers(res, height_range) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Every per-frame output as a BGR image the size of the rectified frame, plus legend text."""
    t, size = res.terrain, (res.left.shape[1], res.left.shape[0])
    cell = lambda a, lo, hi, cmap=cv2.COLORMAP_INFERNO: colourise(a, lo, hi, cmap, size)
    disp = t.disparity
    d_lo, d_hi = (np.nanpercentile(disp, [2, 98]) if np.isfinite(disp).any() else (0.0, 1.0))
    rough_hi = max(float(np.nanpercentile(t.roughness, 95)), 1e-3) if np.isfinite(t.roughness).any() else 0.1
    layers = {"left": cv2.cvtColor(res.left, cv2.COLOR_GRAY2BGR),
              "right": cv2.cvtColor(res.right, cv2.COLOR_GRAY2BGR),
              "disparity": colourise(disp, d_lo, d_hi),
              "height": colourise(t.height, *height_range),
              "slope": cell(t.slope_deg, 0, 60),
              "roughness": cell(t.roughness, 0, rough_hi)}
    legend = {"left": "rectified left camera", "right": "rectified right camera",
              "disparity": f"{d_lo:.1f} .. {d_hi:.1f} px (near = red)",
              "height": f"{height_range[0]:g} .. {height_range[1]:g} m above the ground plane",
              "slope": "0 .. 60 deg", "roughness": f"0 .. {rough_hi:.2f} m"}
    if res.scene is not None:
        s = res.scene
        noise = s.noise if math.isfinite(s.noise) else 1.0
        layers["regions"] = overlay(res.left, regions_bgr(s.regions))
        nov = cell(s.novelty, 0, 6)
        layers["novelty"] = cv2.addWeighted(cv2.cvtColor(res.left, cv2.COLOR_GRAY2BGR), 0.35, nov, 0.65, 0)
        layers["contrast"] = cell(s.contrast, 0, 3 * noise)
        legend |= {"regions": f"{len(s.region_info)} regions, colours as in the table",
                   "novelty": "z-score 0 .. 6 (> 3 unusual; grey = warm-up or no depth)",
                   "contrast": f"0 .. {3 * noise:.2f} (3 x feature noise)"}
    else:
        blank = np.full((size[1], size[0], 3), 40, np.uint8)
        for k in ("regions", "novelty", "contrast"):
            layers[k] = blank
            legend[k] = "needs --model"
    return layers, legend


def frame_extra(res) -> dict:
    t = res.terrain
    out = {"timings": {k: round(v, 1) for k, v in res.timings_ms.items()},
           "plane": None if t.plane is None else [round(float(v), 4) for v in t.plane]}
    if res.scene is not None:
        s = res.scene
        out["noise"] = round(float(s.noise), 3) if math.isfinite(s.noise) else None
        out["regions"] = [{"id": rid, "cells": v["cells"], "weight": round(float(v["weight"]), 3),
                           "height": None if not math.isfinite(v["height"]) else round(v["height"], 2),
                           "colour": "#%02x%02x%02x" % tuple(int(c) for c in PALETTE[rid % len(PALETTE)][::-1])}
                          for rid, v in sorted(s.region_info.items())]
    return out


def render_map(grid, layer: str, path, height_range, scale: int = 4) -> np.ndarray:
    """One map layer, cropped to the explored area, north up, with the flown path and anomalies."""
    seen = grid.seen > 0
    if layer == "regions":
        img = np.where(seen[..., None], PALETTE[np.maximum(grid.regions, 0) % len(PALETTE)], 30).astype(np.uint8)
    elif layer == "anomalies":
        img = colourise(np.where(seen, grid.anomaly_hits, np.nan), 0, 2 * grid.cfg.min_hits, cv2.COLORMAP_INFERNO)
    elif layer == "coverage":
        img = colourise(np.where(seen, np.log1p(grid.seen), np.nan), 0, max(np.log1p(grid.seen.max()), 1), cv2.COLORMAP_VIRIDIS)
    else:
        img = colourise(grid.height, *height_range)
    img = np.ascontiguousarray(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))

    def px(x, y):
        r, c = grid.world_to_cell(x, y)
        return int(c * scale + scale / 2), int(r * scale + scale / 2)

    if len(path) > 1:
        cv2.polylines(img, [np.array([px(x, y) for x, y in path], np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    if path:
        cv2.circle(img, px(*path[-1]), 5, (255, 255, 255), -1)
    for a in grid.anomalies():
        cv2.circle(img, px(a["x"], a["y"]), 10, (0, 0, 255), 2)
    rr, cc = np.nonzero(seen)
    m = 6
    img = img[max(rr.min() - m, 0) * scale: (rr.max() + m + 1) * scale, max(cc.min() - m, 0) * scale: (cc.max() + m + 1) * scale]
    return np.ascontiguousarray(np.flipud(img))  # north (+y) up


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: float(v) if isinstance(v, np.generic) else str(v)))


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            if url.path in ("/", "/index.html"):
                return self._send(HTML.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._send(_state_json(session), "application/json")
            if len(parts) == 4 and parts[:2] == ["api", "frame"] and parts[3].endswith(".jpg"):
                body = session.layer(int(parts[2]), parts[3][:-4])
                return self._send(body, "image/jpeg") if body else self._send(b"", "text/plain", 404)
            if len(parts) == 3 and parts[:2] == ["api", "map"] and parts[2].endswith(".png"):
                body = session.map_image(parts[2][:-4])
                return self._send(body, "image/png") if body else self._send(b"", "text/plain", 404)
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/control":
                return self._send(b"not found", "text/plain", 404)
            q = parse_qs(url.query)
            session.control(q.get("action", [""])[0], q.get("value", [None])[0])
            self._send(_state_json(session), "application/json")

    return Handler


def _state_json(session: Session) -> bytes:
    def clean(v):  # NaN/inf aren't valid JSON
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(session.state())).encode()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_arguments(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--play-fps", type=float, default=0.0, help="limit playback speed (0 = as fast as it computes)")
    ap.add_argument("--keep", type=int, default=300, help="processed frames kept for stepping back")
    args = ap.parse_args(argv)

    session = Session(args)
    session.control("step")  # show the first frame straight away
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"perception viewer on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
