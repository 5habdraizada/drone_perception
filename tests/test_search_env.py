"""Search environment: API, crashes, scoring of detections, and the planners' safety and value."""
import copy

import numpy as np
import pytest

from sim.search_env import DroneSearchEnv, Episode, SearchConfig, random_episode
from sim.search_policies import GreedyInfoPolicy, RandomPolicy, run_episode
from sim.synthetic_stereo import Block, Scene

AREA = (-15.0, 15.0, -10.0, 10.0)


def _episode(*blocks):
    hidden = [b for b in blocks if b.pattern == "hidden"]
    return Episode(Scene(extent=80.0, res=0.1, blocks=list(blocks)), hidden)


def test_reset_and_step_follow_the_gym_api():
    cfg = SearchConfig(area=AREA, max_steps=3, window=16)
    env = DroneSearchEnv(cfg, seed=1)
    obs, info = env.reset(_episode(Block(5, 0, 4, 4, 5, pattern="canopy")))
    assert set(obs) == {"p_object", "view_quality", "height", "searchable", "state"}
    assert obs["p_object"].shape == (16, 16) and obs["state"].shape == (4,)
    assert info["step"] == 0 and info["entropy"] > 0
    for t in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        assert np.isfinite(reward) and not terminated
    assert truncated and info["step"] == 3


def test_flying_low_into_a_roof_ends_the_episode():
    cfg = SearchConfig(area=AREA, step_m=5.0)
    env = DroneSearchEnv(cfg, seed=1)
    env.reset(_episode(Block(-8.0, -8.0, 6, 6, 10.0)))  # start is (-13, -8): the roof is just east
    _, reward, terminated, _, info = env.step(8)        # east, low altitude (8 m < 10 m roof)
    assert terminated and info["crashed"] and reward <= -cfg.crash_penalty


def test_each_real_target_pays_once_and_false_alarms_cost():
    cfg = SearchConfig(area=AREA)
    env = DroneSearchEnv(cfg, seed=1)
    env.reset(_episode(Block(5, 0, 1, 1, 0, pattern="hidden"), Block(-5, 5, 1, 1, 0, pattern="hidden")))
    fake = []
    env.belief.detections = lambda kind="object": fake

    fake[:] = [{"x": 5.5, "y": 0.5}]
    assert env._score_detections() == cfg.discovery_bonus
    assert env._score_detections() == 0.0                   # same detection again: nothing
    fake[:] = [{"x": -10.0, "y": -5.0}]
    assert env._score_detections() == -cfg.false_alarm_penalty
    fake[:] = [{"x": -5.0, "y": 4.0}]
    assert env._score_detections() == cfg.discovery_bonus
    info = env.info()
    assert (info["found"], info["targets"], info["false_alarms"]) == (2, 2, 1)


def test_greedy_planner_never_flies_low_over_unmapped_ground():
    env = DroneSearchEnv(SearchConfig(area=AREA), seed=1)
    env.reset(_episode())
    policy = GreedyInfoPolicy()
    assert policy._known_safe(env, env.x + 5, env.y, 20.0)         # cruise height clears anything
    assert not policy._known_safe(env, env.x, env.y + 15, 8.0)     # low into the unknown: no


@pytest.mark.slow
def test_information_greedy_beats_random_flight():
    cfg = SearchConfig(max_steps=60)
    ep = random_episode(cfg, np.random.default_rng(3))
    greedy = run_episode(DroneSearchEnv(cfg, seed=2), GreedyInfoPolicy(), copy.deepcopy(ep))
    rand = run_episode(DroneSearchEnv(cfg, seed=2), RandomPolicy(1), copy.deepcopy(ep))
    assert not greedy["crashed"]
    assert greedy["reward"] > rand["reward"]
    assert greedy["entropy_removed"] > rand["entropy_removed"]
