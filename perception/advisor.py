"""Mission advisor: a vision-language model's judgement of anomalies against a plain-language brief.

The advisor only changes what the drone looks at: it ranks confirmed anomalies, and can dismiss
ones that don't match the brief. It never touches safety (return, landing, altitude clearance,
battery); those stay geometric in decision.py.

    advisor = AsyncAdvisor(SomeAdvisor(...), brief="Find vehicles and people. Ignore trees.")
    planner = MissionPlanner(home, cfg, advisor=advisor, crops=snapshots.crop)

Models are slow (seconds per query), so AsyncAdvisor runs them in a background thread and the
planner uses an answer only once it has arrived.
"""
from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Assessment:
    label: str          # short description of what the crop shows
    relevance: float    # 0..1: how relevant to the mission brief
    investigate: bool   # worth a closer look
    reason: str
    source: str         # which advisor produced it


class Advisor(Protocol):
    def assess(self, image: np.ndarray, brief: str) -> Assessment:
        """image: BGR or greyscale crop centred on the anomaly."""
        ...


PROMPT = """You are helping an autonomous survey drone decide what to inspect.
Mission brief: {brief}
The image is a top-down aerial crop centred on something the drone found unusual.
Reply with JSON only, no other text:
{{"label": "<what it is, a few words>", "relevance": <0.0 to 1.0, how relevant to the brief>,
 "investigate": <true or false>, "reason": "<one short sentence>"}}"""


def parse_reply(text: str, source: str) -> Assessment:
    """Turn a model's reply into an Assessment, tolerating extra text around the JSON.
    Unparseable replies count as 'unsure, investigate': a model failure must never hide a target."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    if not d:
        return Assessment("unknown", 0.5, True, f"unparseable reply: {text[:80]!r}", source)
    try:
        rel = float(d.get("relevance", 0.5))
    except (TypeError, ValueError):
        rel = 0.5
    rel = 0.5 if not math.isfinite(rel) else min(1.0, max(0.0, rel))
    inv = d.get("investigate", True)
    inv = inv if isinstance(inv, bool) else str(inv).strip().lower() not in ("false", "no", "0")
    return Assessment(str(d.get("label", "unknown"))[:80], rel, inv, str(d.get("reason", ""))[:200], source)


class AsyncAdvisor:
    """Runs an Advisor in a background thread; each anomaly is assessed once."""

    def __init__(self, advisor: Advisor, brief: str):
        self.advisor, self.brief = advisor, brief
        self._results: dict[str, Assessment] = {}
        self._pending: set[str] = set()
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def request(self, key: str, image: np.ndarray) -> None:
        with self._lock:
            if key in self._results or key in self._pending:
                return
            self._pending.add(key)
        self._jobs.put((key, image))

    def result(self, key: str) -> Assessment | None:
        with self._lock:
            return self._results.get(key)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def wait(self, timeout: float = 30.0) -> None:
        """Block until queued assessments are done (for tests and offline runs)."""
        end = time.monotonic() + timeout
        while self.busy() and time.monotonic() < end:
            time.sleep(0.01)

    def _worker(self) -> None:
        while True:
            key, image = self._jobs.get()
            try:
                result = self.advisor.assess(image, self.brief)
            except Exception as e:  # the advisor failing must not hide the anomaly
                result = Assessment("unknown", 0.5, True, f"advisor error: {type(e).__name__}: {e}", "error")
            with self._lock:
                self._results[key] = result
                self._pending.discard(key)


class Snapshots:
    """Recent camera frames with poses, to crop the best view of a world point for the advisor."""

    def __init__(self, k: np.ndarray, keep: int = 200):
        self.k = np.asarray(k, np.float64)
        self.frames: deque = deque(maxlen=keep)

    def add(self, image: np.ndarray, t_world_cam: np.ndarray) -> None:
        self.frames.append((image, np.asarray(t_world_cam, np.float64)))

    def crop(self, x: float, y: float, z: float = 0.0, size_m: float = 6.0, out_px: int = 384) -> np.ndarray | None:
        """Square crop around world point (x, y, z), size_m across, from the frame that saw it
        closest to the image centre; None if no stored frame saw it."""
        best, best_off = None, math.inf
        p_w = np.array([x, y, z, 1.0])
        for image, pose in self.frames:
            p_c = np.linalg.inv(pose) @ p_w
            if p_c[2] <= 0.1:
                continue
            u, v = (self.k @ (p_c[:3] / p_c[2]))[:2]
            h, w = image.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                continue
            off = math.hypot(u - w / 2, v - h / 2)
            if off < best_off:
                best, best_off = (image, u, v, p_c[2]), off
        if best is None:
            return None
        image, u, v, depth = best
        half = max(8, int(round(size_m * self.k[0, 0] / depth / 2)))
        pad = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REPLICATE)
        r, c = int(round(v)) + half, int(round(u)) + half
        crop = pad[r - half: r + half, c - half: c + half]
        return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_LINEAR)


class FunctionAdvisor:
    """Wraps a plain function as an Advisor; used for tests and as a stand-in before a model is set up."""

    def __init__(self, fn: Callable[[np.ndarray, str], tuple[str, float, bool, str]], name: str = "function"):
        self.fn, self.name = fn, name

    def assess(self, image: np.ndarray, brief: str) -> Assessment:
        label, relevance, investigate, reason = self.fn(image, brief)
        return Assessment(label, float(relevance), bool(investigate), reason, self.name)
