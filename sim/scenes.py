"""Shared synthetic setups for the demos, sample data and tests."""
from __future__ import annotations

from perception.terrain import StereoRig
from sim.synthetic_stereo import Block, Scene

# A rectified 640 × 352 pair like the real rig produces after calibration.
FRAME = (640, 352)
RIG = StereoRig(fx=577.0, fy=577.0, cx=FRAME[0] / 2, cy=FRAME[1] / 2, baseline=0.10)


def flight_scene() -> Scene:
    """30 m square: four roofs, a pit, a painted strip along y = -1.2 and one striped object."""
    return Scene(extent=30.0, seed=3, blocks=[
        Block(-8.0, 0.6, 2.0, 1.6, 1.8),
        Block(-4.5, -0.8, 1.6, 1.4, 1.2),
        Block(-1.0, 0.7, 2.2, 1.2, 2.2),
        Block(2.5, -0.6, 1.8, 1.6, 1.5),
        Block(-6.0, -1.2, 1.2, 0.8, -0.7),
        Block(0.0, -1.2, 30.0, 0.7, 0.0),
        Block(7.5, 0.5, 0.9, 0.9, 0.5, pattern="stripes"),
    ])
