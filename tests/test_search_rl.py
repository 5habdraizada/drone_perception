"""Learned search policy: features, masking of unsafe moves, and a policy that runs in NumPy only."""
import numpy as np

from sim.search_env import DroneSearchEnv, Episode, SearchConfig
from sim.search_rl import N_FEATURES, LearnedPolicy, action_features, forward, init_params
from sim.synthetic_stereo import Block, Scene

AREA = (-15.0, 15.0, -10.0, 10.0)


def _env(*blocks):
    env = DroneSearchEnv(SearchConfig(area=AREA, max_steps=5), seed=1)
    hidden = [b for b in blocks if b.pattern == "hidden"]
    env.reset(Episode(Scene(extent=80.0, res=0.1, blocks=list(blocks)), hidden))
    return env


def test_features_cover_every_move_and_low_moves_into_the_unknown_are_unsafe():
    env = _env(Block(5, 0, 6, 6, 5, pattern="canopy"))
    feats, safe = action_features(env)
    assert feats.shape == (16, N_FEATURES) and np.isfinite(feats).all()
    assert safe[:8].any()                    # cruise moves
    assert not safe[8:].all()                # low moves over unmapped ground are masked


def test_policy_never_picks_a_masked_move():
    params = init_params(np.random.default_rng(0))
    feats = np.random.default_rng(1).normal(size=(16, N_FEATURES))
    safe = np.zeros(16, bool)
    safe[[2, 5]] = True
    logp = forward(params, feats, safe)
    assert np.isclose(np.exp(logp).sum(), 1.0)
    assert np.exp(logp)[~safe].max() < 1e-12


def test_learned_policy_flies_an_episode():
    env = _env(Block(5, 0, 6, 6, 5, pattern="canopy"), Block(5, 0, 1, 1, 0, pattern="hidden"))
    policy = LearnedPolicy(init_params(np.random.default_rng(0)), sample=True, seed=3)
    done = trunc = False
    while not (done or trunc):
        _, _, done, trunc, info = env.step(policy(env))
    assert not info["crashed"] and info["step"] == 5
