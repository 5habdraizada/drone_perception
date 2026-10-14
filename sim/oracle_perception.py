"""Exact perception frames straight from a synthetic scene, for fast closed-loop mission tests.

Rendering stereo pairs and running the real pipeline takes seconds per frame;
whole missions need hundreds of frames. This produces what a perfect
terrain + scene pipeline would output for a camera looking straight down:
cell heights from ray casting against the scene's blocks (walls give no data),
a region per block, roughness (canopy is rough), and high novelty on unusual
objects; objects hidden under canopy are only glimpsed now and then. Optional
random one-frame false alarms check that the map and planner don't chase noise.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from perception.terrain import StereoRig
from sim.synthetic_stereo import Scene

_R_DOWN = np.diag([1.0, -1.0, -1.0])  # camera x = world x, camera y = world -y, looking down


class OraclePerception:
    def __init__(self, scene: Scene, rig: StereoRig, image_size=(640, 352), grid=(20, 11),
                 novel_z: float = 6.0, false_alarm_rate: float = 0.0, seed: int = 0):
        """false_alarm_rate: chance per frame of one random cell scoring novel_z for that frame only.

        Block patterns: "stripes" (an unusual object), "canopy" (tree canopy: rough, hides the
        ground), "hidden" (an object on the ground, glimpsed under canopy with probability
        glimpse_low below low_agl metres, glimpse_high above)."""
        self.glimpse_low, self.glimpse_high, self.low_agl = 0.6, 0.1, 12.0
        self.scene, self.rig = scene, rig
        self.novel_z, self.false_alarm_rate = novel_z, false_alarm_rate
        self.rng = np.random.default_rng(seed)
        gw, gh = grid
        u = (np.arange(gw) + 0.5) * image_size[0] / gw
        v = (np.arange(gh) + 0.5) * image_size[1] / gh
        uu, vv = np.meshgrid(u, v)
        self.rays = np.stack([(uu - rig.cx) / rig.fx, (vv - rig.cy) / rig.fy, np.ones_like(uu)], -1) @ _R_DOWN.T

    @staticmethod
    def pose(x: float, y: float, agl: float) -> np.ndarray:
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = _R_DOWN, (x, y, agl)
        return t

    def frame(self, x: float, y: float, agl: float, t: float):
        """SceneFrame-like object (regions, novelty, terrain.plane, terrain.cell_height) seen from (x, y, agl)."""
        centre = np.array([x, y, agl])
        gh, gw = self.rays.shape[:2]
        height = np.full((gh, gw), np.nan)
        regions = np.zeros((gh, gw), int)
        novelty = np.zeros((gh, gw))
        done = np.zeros((gh, gw), bool)

        rough = np.full((gh, gw), 0.02)
        canopy = np.zeros((gh, gw), bool)
        hidden = [kb for kb in enumerate(self.scene.blocks) if kb[1].pattern == "hidden"]
        blocks = sorted((kb for kb in enumerate(self.scene.blocks) if kb[1].pattern != "hidden"),
                        key=lambda kb: -kb[1].h)
        # Structures, tallest first: a ray takes the first top it meets.
        for k, b in blocks:
            if b.h <= 0:
                continue
            hit = ~done & self._inside(b, centre + (agl - b.h) * self.rays)
            height[hit], regions[hit] = b.h, k + 1
            done |= hit
            if b.pattern == "stripes":
                novelty[hit] = self.novel_z
            if b.pattern == "canopy":
                rough[hit], canopy = 0.3, canopy | hit
        # Ground, unless the ground point is inside a block's footprint (then it's a wall, or a pit).
        ground = centre + agl * self.rays
        in_any = np.zeros((gh, gw), bool)
        for _, b in blocks:
            if b.h != 0:
                in_any |= self._inside(b, ground)
        flat = ~done & ~in_any
        height[flat] = 0.0
        done |= flat
        for k, b in blocks:  # painted flat areas keep their own region
            if b.h == 0:
                paint = flat & self._inside(b, ground)
                regions[paint] = k + 1
                if b.pattern == "stripes":
                    novelty[paint] = self.novel_z
        for k, b in blocks:  # pit floors
            if b.h < 0:
                hit = ~done & self._inside(b, centre + (agl - b.h) * self.rays)
                height[hit], regions[hit] = b.h, k + 1
                done |= hit

        # Objects on the ground: in the open they're plainly unusual; under canopy they're only
        # glimpsed through gaps, far more often on a low pass.
        p_glimpse = self.glimpse_low if agl < self.low_agl else self.glimpse_high
        for k, b in hidden:
            under = self._inside(b, ground)
            novelty[under & flat] = self.novel_z
            seen = under & canopy & (self.rng.random((gh, gw)) < p_glimpse)
            novelty[seen] = self.novel_z

        if self.rng.random() < self.false_alarm_rate:
            novelty[self.rng.integers(gh), self.rng.integers(gw)] = self.novel_z
        terrain = SimpleNamespace(plane=np.array([0.0, 0.0, -1.0, agl]), cell_height=height, roughness=rough)
        return SimpleNamespace(t=t, regions=regions, novelty=novelty, terrain=terrain)

    @staticmethod
    def _inside(b, p) -> np.ndarray:
        return (np.abs(p[..., 0] - b.x) < b.w / 2) & (np.abs(p[..., 1] - b.y) < b.l / 2)
