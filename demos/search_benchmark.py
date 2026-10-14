"""Is the reward meaningful? Compare search policies on random scenes with hidden targets.

    python -m demos.search_benchmark [--episodes 8] [--steps 150] [--out out/search_demo.png]

Each episode: roofs and tree canopy hiding objects on the ground (sim/search_env.py).
Policies (sim/search_policies.py):
  random      random safe moves
  lawnmower   systematic sweep at cruise altitude: the usual coverage baseline
  greedy      each step, the move with the highest expected information gain (BeliefMap.expected_gain)
              minus energy: what the reward asks for, without any learning
Draws the greedy run of the first episode: flown path (white high, cyan low), the object belief,
and the true hidden objects (green).
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import cv2
import numpy as np

from sim.search_env import DroneSearchEnv, SearchConfig, random_episode
from sim.search_policies import GreedyInfoPolicy, LawnmowerPolicy, RandomPolicy, run_episode

POLICIES = {"random": lambda: RandomPolicy(1), "lawnmower": LawnmowerPolicy, "greedy": GreedyInfoPolicy}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/search_demo.png")
    args = ap.parse_args(argv)

    cfg = SearchConfig(max_steps=args.steps)
    rng = np.random.default_rng(args.seed)
    episodes = [random_episode(cfg, rng) for _ in range(args.episodes)]

    print(f"{args.episodes} episodes x {args.steps} steps; totals over all episodes\n")
    print(f"{'policy':10s} {'reward':>8s} {'entropy':>8s} {'found':>7s} {'false':>6s} "
          f"{'crashes':>7s} {'flown':>7s} {'s/ep':>5s}")
    drawn = None
    for name, make in POLICIES.items():
        rows = []
        t0 = time.perf_counter()
        for i, ep in enumerate(episodes):
            env = DroneSearchEnv(cfg, seed=100 + i)
            rows.append(run_episode(env, make(), copy.deepcopy(ep)))
            if name == "greedy" and i == 0:
                drawn = (env, rows[-1])
        tot = lambda k: sum(r[k] for r in rows)
        print(f"{name:10s} {np.mean([r['reward'] for r in rows]):8.1f} "
              f"{np.mean([r['entropy_removed'] for r in rows]):8.0%} "
              f"{tot('found'):>3d}/{tot('targets'):<3d} "
              f"{tot('false_alarms'):6d} {tot('crashed'):7d} "
              f"{np.mean([r['flown_m'] for r in rows]):6.0f}m {(time.perf_counter() - t0) / len(rows):5.1f}")
    print("\nreward and entropy: mean per episode (entropy = share of the starting uncertainty removed)")

    if drawn is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw(*drawn))
        print(f"greedy run of episode 1 drawn to {out}")


def draw(env: DroneSearchEnv, result: dict, scale: int = 8) -> np.ndarray:
    g, b, c = env.grid, env.belief, env.cfg
    r0, c0 = g.world_to_cell(c.area[0], c.area[2])
    r1, c1 = g.world_to_cell(c.area[1], c.area[3])
    rs, cs = slice(r0, r1), slice(c0, c1)
    top_y = g.cell_to_world(r1, c0)[1] - g.cfg.resolution / 2  # northern edge of the drawn cells
    left_x = g.cell_to_world(r0, c0)[0] - g.cfg.resolution / 2

    def panel(layer, cmap, title):
        img = cv2.applyColorMap(np.uint8(np.clip(layer[rs, cs], 0, 1) * 255), cmap)
        img = cv2.flip(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), 0)  # north up
        to_px = lambda x, y: (int((x - left_x) / g.cfg.resolution * scale),
                              int((top_y - y) / g.cfg.resolution * scale))
        for (x0, y0, a0), (x1, y1, a1) in zip(result["path"], result["path"][1:]):
            cv2.line(img, to_px(x0, y0), to_px(x1, y1), (255, 255, 0) if a1 < c.altitudes[0] else (255, 255, 255), 2)
        for h in env.ep.hidden:
            cv2.circle(img, to_px(h.x, h.y), 9, (0, 255, 0), 2)
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    heights = np.nan_to_num(g.height, nan=0.0) / 12.0
    p_obj = b.p_object / max(float(b.p_object[rs, cs].max()), 1e-6)
    return np.hstack([panel(heights, cv2.COLORMAP_BONE, "map heights + path"),
                      panel(p_obj, cv2.COLORMAP_INFERNO, "P(hidden object), scaled")])


if __name__ == "__main__":
    main()
