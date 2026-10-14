"""A learned search policy for DroneSearchEnv: per-move features scored by a small network.

The policy sees the same belief map the greedy planner uses, but more of it. For each of the 16
moves it gets the expected information gain right there (what greedy maximises), plus how much
uncertainty lies further along that direction, split into ground under canopy and open ground.
That lookahead lets it commit to a low pass over trees that only pays off a few steps later.

One small network scores every move from its features (the same weights for every move), and
moves that aren't known to be safe are masked out. Inference is plain NumPy, so the drone needs
no deep-learning framework; training (tools/train_search_policy.py) uses PyTorch.
"""
from __future__ import annotations

import math

import numpy as np

from perception.belief import _entropy
from sim.search_env import DIRECTIONS, DroneSearchEnv
from sim.search_policies import GreedyInfoPolicy, _target

N_FEATURES = 13
_SAFETY = GreedyInfoPolicy()


def action_features(env: DroneSearchEnv) -> tuple[np.ndarray, np.ndarray]:
    """(features [16, N_FEATURES], safe mask [16]) for the drone's current state."""
    cfg, g, b = env.cfg, env.grid, env.belief
    rr, cc = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    cx, cy = g.cell_to_world(rr.ravel(), cc.ravel())
    h = (_entropy(b.p_object) * b.searchable * g.area_mask(*cfg.area)).ravel()
    canopy = b._canopy_cells(np.arange(h.size))
    remaining = env._entropy / max(env.initial_entropy, 1e-9)

    feats = np.zeros((env.n_actions, N_FEATURES))
    safe = np.zeros(env.n_actions, bool)
    ahead = {}
    for k, (dx, dy) in enumerate(DIRECTIONS):  # uncertainty further along each direction, from its target
        nx, ny, _ = _target(env, k)
        along = (cx - nx) * dx + (cy - ny) * dy
        side = np.abs(-(cx - nx) * dy + (cy - ny) * dx)
        near = (along > -2) & (along <= 12) & (side <= 6)
        far = (along > 12) & (along <= 30) & (side <= 8)
        below = (cx - nx) ** 2 + (cy - ny) ** 2 <= 16
        ahead[k] = [h[near & canopy].sum(), h[near & ~canopy].sum(), h[far & canopy].sum(),
                    h[far & ~canopy].sum(), h[below & canopy].sum()]
    for a in range(env.n_actions):
        nx, ny, alt = _target(env, a)
        moved = math.hypot(nx - env.x, ny - env.y)
        dist = moved + abs(alt - env.agl)
        safe[a] = _SAFETY._known_safe(env, nx, ny, alt)
        feats[a] = [b.expected_gain(nx, ny, alt) / 10, dist / 20, a // 8, env.alt_i, env.t / cfg.max_steps,
                    moved / cfg.step_m, remaining, *(np.array(ahead[a % 8]) / 20), 1.0]
    if not safe.any():
        safe[:8] = True  # cruise altitude clears anything the map could hold
    return feats, safe


def init_params(rng: np.random.Generator, hidden: int = 48) -> dict:
    s = lambda n_in: 1.0 / math.sqrt(n_in)
    return {"w1": rng.normal(0, s(N_FEATURES), (N_FEATURES, hidden)), "b1": np.zeros(hidden),
            "w2": rng.normal(0, s(hidden), (hidden, hidden)), "b2": np.zeros(hidden),
            "w3": rng.normal(0, 0.01, (hidden, 1)), "b3": np.zeros(1)}


def forward(params: dict, feats: np.ndarray, safe: np.ndarray) -> np.ndarray:
    """Log-probabilities over moves (unsafe ones get none)."""
    h1 = np.tanh(feats @ params["w1"] + params["b1"])
    h2 = np.tanh(h1 @ params["w2"] + params["b2"])
    logits = (h2 @ params["w3"] + params["b3"]).ravel()
    logits = np.where(safe, logits, -1e9)
    logp = logits - logits.max()
    return logp - np.log(np.exp(logp).sum())


class LearnedPolicy:
    """Picks moves with a trained network. sample=True draws from its distribution (for training)."""

    def __init__(self, params: dict | str, sample: bool = False, seed: int = 0):
        if isinstance(params, str):
            params = dict(np.load(params))
        self.params, self.sample = params, sample
        self.rng = np.random.default_rng(seed)
        self.last = None  # (features, safe, action, log-prob) of the latest decision

    def __call__(self, env: DroneSearchEnv) -> int:
        feats, safe = action_features(env)
        logp = forward(self.params, feats, safe)
        a = int(self.rng.choice(len(logp), p=np.exp(logp))) if self.sample else int(np.argmax(logp))
        self.last = (feats, safe, a, float(logp[a]))
        return a
