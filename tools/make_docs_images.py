"""Regenerate the images in docs/images used by the README's walkthrough.

    python -m tools.make_docs_images [--out docs/images] [--skip-real]

Everything is synthetic with exact ground truth, except the Delft figure, which needs
data/delft from tools/fetch_pdok.py (skipped if missing, or with --skip-real).
The search figure needs models/search_policy.npz from tools/train_search_policy.py.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((24, img.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (6, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit_width(img: np.ndarray, width: int) -> np.ndarray:
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)


def rectification(out: Path) -> None:
    """Raw, distorted, mismatched-lens pair vs the rectified pair, with matching rows marked."""
    from perception.calibration import Rectifier
    from sim.scenes import flight_scene
    from sim.synthetic_stereo import camera_pose, render_view
    from tools.make_sample_data import RAW, sample_calibration

    calib = sample_calibration()
    r_wc, c = camera_pose(6.0, 1.0, -0.5, (-3.0, 0.0))
    rng = np.random.default_rng(0)
    left, _, _ = render_view(flight_scene(), calib.left.K, calib.left.dist, RAW, r_wc, c, rng=rng)
    right, _, _ = render_view(flight_scene(), calib.right.K, calib.right.dist, RAW, r_wc @ calib.R.T,
                              c + r_wc @ (-calib.R.T @ calib.T), rng=rng)
    rl, rr = Rectifier(calib).rectify(left, right)
    raw = np.hstack([fit_width(left, 640), fit_width(right, 640)])
    rect = cv2.cvtColor(np.hstack([rl, rr]), cv2.COLOR_GRAY2BGR) if rl.ndim == 2 else np.hstack([rl, rr])
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw
    for y in range(20, rect.shape[0], 40):
        cv2.line(rect, (0, y), (rect.shape[1], y), (80, 220, 80), 1)
    img = np.vstack([label(raw, "raw: long lens (left) and wide lens (right), different distortion and rotation"),
                     label(rect, "rectified: one shared 640 x 352 pinhole camera, matching points on the same row")])
    cv2.imwrite(str(out / "01_rectification.png"), img)


def demos(out: Path) -> None:
    from demos import map as map_demo
    from demos import mission, scene, terrain

    terrain.main(["--out", str(out / "02_terrain.png")])
    with tempfile.TemporaryDirectory() as tmp:
        scene.main(["--out", f"{tmp}/scene.png"])
        img = cv2.imread(f"{tmp}/scene.png")
        cv2.imwrite(str(out / "03_scene.png"), fit_width(img, 960))
    map_demo.main(["--out", str(out / "04_map.png")])
    mission.main(["--out", str(out / "06_mission.png")])
    mission.main(["--search", "--out", str(out / "07_mission_search.png")])


def belief(out: Path) -> None:
    """Trees hiding an object, a roof and open ground: what high and low passes do to the beliefs."""
    from perception.belief import BeliefMap
    from perception.mapping import GridMap, MapConfig
    from sim.oracle_perception import OraclePerception
    from sim.synthetic_stereo import Block, Scene
    from sim.search_env import RIG, SIZE

    trees, roof = Block(6.0, 0.0, 7.0, 7.0, 5.0, pattern="canopy"), Block(-7.0, 1.0, 6.0, 5.0, 6.0)
    hidden = Block(6.5, 1.0, 1.0, 1.0, 0.0, pattern="hidden")
    oracle = OraclePerception(Scene(extent=60.0, res=0.1, blocks=[trees, roof, hidden]), RIG, SIZE, seed=2)
    grid = GridMap(RIG, SIZE, MapConfig(resolution=0.5))
    area = (-16.0, 16.0, -8.0, 8.0)
    grid.ensure_area(*area)
    b = BeliefMap(grid, RIG, SIZE)

    def fly(xs, agl, y=0.0):
        for i, x in enumerate(xs):
            f, pose = oracle.frame(x, y, agl, float(i)), oracle.pose(x, y, agl)
            grid.update(f, pose)
            b.observe(f, pose)

    def panel(layer, lo, hi, cmap, title, path=()):
        r0, c0 = grid.world_to_cell(area[0], area[2])  # the map grows as it flies: index it now
        r1, c1 = grid.world_to_cell(area[1], area[3])
        img = cv2.applyColorMap(np.uint8(np.clip((layer[r0:r1, c0:c1] - lo) / (hi - lo), 0, 1) * 255), cmap)
        img = cv2.flip(cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST), 0)
        px = lambda x, y: (int((x - area[0]) / 0.5 * 8), int((area[3] - y) / 0.5 * 8))
        for (xa, ya, col) in path:
            cv2.line(img, px(xa[0], ya), px(xa[1], ya), col, 2)
        cv2.circle(img, px(hidden.x, hidden.y), 14, (0, 255, 0), 2)
        return label(img, title)

    high = ((-14, 14), 0.0, (255, 255, 255))
    fly(np.arange(-14, 15, 2.0), agl=20)
    after_high = [panel(b.view_quality, 0, 1, cv2.COLORMAP_VIRIDIS, "view quality after a high pass (20 m)", [high]),
                  panel(np.log10(b.p_object + 1e-4), -3, 0, cv2.COLORMAP_INFERNO, "P(hidden object), log scale, after the high pass", [high])]
    fly(np.arange(2, 11, 0.5), agl=8)
    low = ((2, 10), 0.0, (255, 255, 0))
    heights = panel(np.nan_to_num(grid.height, nan=0.0), 0, 6, cv2.COLORMAP_BONE,
                    "mapped heights: roof (left), trees (right)", [high, low])
    rows = [np.hstack([heights, after_high[0]]),
            np.hstack([after_high[1],
                       panel(np.log10(b.p_object + 1e-4), -3, 0, cv2.COLORMAP_INFERNO, "... after a low pass over the trees (8 m)", [high, low])])]
    img = np.vstack(rows)
    found = b.detections("object")
    note = f"found: {', '.join(f'({d['x']:.1f}, {d['y']:.1f})' for d in found) or 'nothing'}; truth ({hidden.x}, {hidden.y})"
    cv2.imwrite(str(out / "05_belief.png"), label(img, "green circle: object hidden under the canopy.   " + note))


def search(out: Path, policy_path: Path, scene_seed: int = 20_005) -> None:
    """Lawnmower, greedy and learned policies on the same held-out scene (an example, not an average)."""
    from sim.search_env import DroneSearchEnv, SearchConfig, random_episode
    from sim.search_policies import GreedyInfoPolicy, LawnmowerPolicy, run_episode
    from sim.search_rl import LearnedPolicy

    cfg = SearchConfig()
    ep = random_episode(cfg, np.random.default_rng(scene_seed))
    policies = [("lawnmower", LawnmowerPolicy()), ("greedy", GreedyInfoPolicy())]
    if policy_path.exists():
        policies.append(("learned (RL)", LearnedPolicy(str(policy_path))))
    panels = []
    for name, pol in policies:
        env = DroneSearchEnv(cfg, seed=7)
        res = run_episode(env, pol, copy.deepcopy(ep))
        g, b = env.grid, env.belief
        r0, c0 = g.world_to_cell(cfg.area[0], cfg.area[2])
        r1, c1 = g.world_to_cell(cfg.area[1], cfg.area[3])
        heights = np.nan_to_num(g.height[r0:r1, c0:c1], nan=0.0) / 12.0
        p = b.p_object[r0:r1, c0:c1]
        base = cv2.applyColorMap(np.uint8(np.clip(heights, 0, 1) * 160), cv2.COLORMAP_BONE)
        heat = cv2.applyColorMap(np.uint8(np.clip(p / 0.05, 0, 1) * 255), cv2.COLORMAP_INFERNO)
        img = cv2.addWeighted(base, 0.5, heat, 0.5, 0)
        s = 6
        img = cv2.flip(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST), 0)
        left, top = g.cell_to_world(r0, c0)[0] - 0.5, g.cell_to_world(r1, c0)[1] - 0.5
        px = lambda x, y: (int((x - left) * s), int((top - y) * s))
        for (x0, y0, _), (x1, y1, a1) in zip(res["path"], res["path"][1:]):
            cv2.line(img, px(x0, y0), px(x1, y1), (255, 255, 0) if a1 < cfg.altitudes[0] else (255, 255, 255), 2)
        for h in env.ep.hidden:
            cv2.circle(img, px(h.x, h.y), 12, (0, 255, 0), 2)
        panels.append(label(img, f"{name}: found {res['found']}/{res['targets']}, reward {res['reward']:.0f}"))
    img = np.hstack(panels)
    cv2.imwrite(str(out / "08_search_policies.png"), label(
        img, f"held-out scene {scene_seed}: path (white 20 m, cyan 8 m), green = hidden object, purple = P(object) still high"))
    curve = policy_path.with_suffix(".png")
    if curve.exists():
        shutil.copy(curve, out / "09_training.png")


def delft(out: Path, area: Path) -> None:
    """Real place: orthophoto, LiDAR truth and what stereo mapped (demos/real_area.py)."""
    from demos import real_area

    with tempfile.TemporaryDirectory() as tmp:
        real_area.main(["--area", str(area), "--baselines", "0.5", "--out", "out/real_area"])
    heights = cv2.imread("out/real_area/heights.png")
    renders = sorted(Path("out/real_area/renders").glob("*_left.png"))
    frame = cv2.imread(str(renders[len(renders) // 2])) if renders else None
    parts = [fit_width(heights, 1200)]
    if frame is not None:
        parts.insert(0, label(fit_width(frame, 1200)[:300], "one rendered camera frame over Delft, 45 m up"))
    cv2.imwrite(str(out / "10_delft.png"), np.vstack(parts))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--policy", default="models/search_policy.npz")
    ap.add_argument("--area", default="data/delft")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset: rectification demos belief search delft")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"rectification": lambda: rectification(out), "demos": lambda: demos(out),
            "belief": lambda: belief(out), "search": lambda: search(out, Path(args.policy)),
            "delft": lambda: delft(out, Path(args.area))}
    for name, job in jobs.items():
        if args.only and name not in args.only:
            continue
        if name == "delft" and (args.skip_real or not Path(args.area).exists()):
            continue
        print(f"== {name}", flush=True)
        job()
    for png in out.glob("*.png"):  # photo-like figures are much smaller as JPEG
        if png.stat().st_size > 300_000:
            cv2.imwrite(str(png.with_suffix(".jpg")), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 88])
            png.unlink()
    print(f"images in {out}/")


if __name__ == "__main__":
    main()
