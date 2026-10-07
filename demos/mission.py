"""A whole autonomous mission in simulation: take off, explore, investigate, come home, land.

    python -m demos.mission [--battery 1.0] [--false-alarms 0.1] [--out out/mission_demo.png]

Uses oracle perception (exact frames from the scene) so a mission takes seconds;
the real pipeline is exercised by demos/map.py. Prints every change of
mode with its reason, the investigation report, and draws the flown path on the map.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from perception.decision import DecisionConfig, MissionPlanner
from perception.mapping import GridMap
from sim.mission import DroneSim, run_mission
from sim.oracle_perception import OraclePerception
from sim.scenes import FRAME, RIG, flight_scene

COLOURS = {"explore": (200, 200, 200), "verify": (0, 200, 255), "investigate": (0, 0, 255),
           "return": (255, 120, 0), "land": (0, 255, 0)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--battery", type=float, default=1.0)
    ap.add_argument("--false-alarms", type=float, default=0.1, help="chance per frame of a one-frame false alarm")
    ap.add_argument("--out", default="out/mission_demo.png")
    args = ap.parse_args(argv)

    scene = flight_scene()
    area = (-12.0, 12.0, -6.0, 6.0)
    perception = OraclePerception(scene, RIG, FRAME, false_alarm_rate=args.false_alarms, seed=2)
    grid = GridMap(RIG, FRAME)
    planner = MissionPlanner(home=(-11.0, -5.0), cfg=DecisionConfig(area=area))
    drone = DroneSim(-11.0, -5.0, battery=args.battery)
    res = run_mission(perception, grid, planner, drone)

    print("mode changes:")
    last = None
    for s in res.trace:
        if s["mode"] != last:
            print(f"  t={s['t']:5.0f} s  ({s['x']:+6.1f}, {s['y']:+5.1f}) {s['agl']:4.1f} m  "
                  f"battery {s['battery']:.0%}  {s['mode']:12s} {s['reason']}")
            last = s["mode"]
    flown = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(res.trace, res.trace[1:]))
    print(f"\n{'landed' if res.landed else 'did NOT land'} after {res.trace[-1]['t']:.0f} s, {flown:.0f} m flown, "
          f"battery {res.trace[-1]['battery']:.0%}")
    odd = next(b for b in scene.blocks if b.pattern == "stripes")
    print(f"investigated {len(planner.report)} anomal{'y' if len(planner.report) == 1 else 'ies'} "
          f"(striped object is at ({odd.x:+.1f}, {odd.y:+.1f})):")
    for a in planner.report:
        print(f"  ({a['x']:+.1f}, {a['y']:+.1f}) height {a['height']:.2f} m, peak z {a['peak_novelty']:.1f}, "
              f"hits {a['hits']}, at t={a['investigated_at']:.0f} s")

    # Height map with the flown path, coloured by mode.
    scale = 6
    h = grid.height
    norm = np.clip((np.nan_to_num(h, nan=-1.0) + 1.0) / 3.5, 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(h)] = 30
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        r, c = grid.world_to_cell(x, y)
        return int(c * scale + scale / 2), int(r * scale + scale / 2)

    for a, b in zip(res.trace, res.trace[1:]):
        cv2.line(img, px(a["x"], a["y"]), px(b["x"], b["y"]), COLOURS[b["mode"]], 2)
    for a in planner.report:
        cv2.circle(img, px(a["x"], a["y"]), 18, (0, 0, 255), 2)
    x0, x1, y0, y1 = area
    cv2.rectangle(img, px(x0, y0), px(x1, y1), (255, 255, 255), 1)
    img = np.flipud(img).copy()  # north up
    r0, c0 = grid.world_to_cell(x0 - 2, y1 + 2)
    r1, c1 = grid.world_to_cell(x1 + 2, y0 - 2)
    rows = img.shape[0]
    img = img[max(rows - (r0 + 1) * scale, 0): rows - r1 * scale, max(c0 * scale, 0): c1 * scale]
    y = 18
    for mode, col in COLOURS.items():
        cv2.putText(img, mode, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y += 18
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
