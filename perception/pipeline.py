"""The full perception chain for one stereo pair: rectify -> terrain -> scene -> map."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from perception.calibration import Rectifier, StereoCalibration
from perception.mapping import GridMap, MapConfig, rectified_camera
from perception.scene import SceneConfig, SceneFrame, SceneModel
from perception.terrain import StereoRig, Terrain, TerrainConfig, TerrainEstimator


@dataclass
class FrameResult:
    t: float
    left: np.ndarray                 # rectified left image (grayscale)
    terrain: Terrain
    scene: SceneFrame | None         # None without a backbone
    mapped: bool                     # added to the map (needs a pose and a ground plane)
    timings_ms: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, rig: StereoRig, image_size: tuple[int, int], rectifier: Rectifier | None = None,
                 backbone_path: str | Path | None = None, terrain_config: TerrainConfig | None = None,
                 scene_config: SceneConfig | None = None, map_config: MapConfig | None = None):
        """rig/image_size describe the rectified images. Without a rectifier, inputs must already be
        rectified. Without a backbone there are no regions or novelty; the map still gets heights."""
        self.rig, self.image_size, self.rectifier = rig, tuple(image_size), rectifier
        self.terrain = TerrainEstimator(rig, terrain_config or TerrainConfig.for_width(image_size[0]))
        backbone = None
        if backbone_path is not None:
            from perception.backbone import Backbone
            backbone = Backbone(backbone_path)
        self.scene = SceneModel(scene_config, backbone) if backbone is not None else None
        self.map = GridMap(rig, image_size, map_config)
        self._plane = None

    @classmethod
    def from_calibration(cls, path: str | Path, width: int = 640, **kw) -> "Pipeline":
        rect = Rectifier(StereoCalibration.load(path), (width, None))
        return cls(rect.rig, rect.out_size, rectifier=rect, **kw)

    def camera_pose(self, t_world_left: np.ndarray) -> np.ndarray:
        """World pose of the raw left camera -> pose of the rectified frame the pipeline works in."""
        return rectified_camera(t_world_left, self.rectifier.R1) if self.rectifier else t_world_left

    def process(self, t: float, left: np.ndarray, right: np.ndarray,
                t_world_left: np.ndarray | None = None, pose_cov: np.ndarray | None = None) -> FrameResult:
        """t_world_left: 4 × 4 world-from-camera pose of the (raw) left camera, if known."""
        t0 = time.perf_counter()
        if self.rectifier is not None:
            left, right = self.rectifier.rectify(left, right)
        left, right = _gray(left), _gray(right)
        if (left.shape[1], left.shape[0]) != self.image_size:
            raise ValueError(f"rectified images are {left.shape[1]}x{left.shape[0]}, expected "
                             f"{self.image_size[0]}x{self.image_size[1]}")
        t1 = time.perf_counter()
        terrain = self.terrain.process(left, right, plane_prior=self._plane)
        self._plane = terrain.plane if terrain.plane is not None else self._plane
        t2 = time.perf_counter()
        scene = self.scene.process(t, left, terrain) if self.scene is not None else None
        t3 = time.perf_counter()

        mapped = False
        if t_world_left is not None and terrain.plane is not None:
            frame = scene if scene is not None else SimpleNamespace(
                t=t, terrain=terrain, regions=np.zeros(terrain.cell_height.shape, int),
                novelty=np.full(terrain.cell_height.shape, np.nan))
            mapped = self.map.update(frame, self.camera_pose(t_world_left), pose_cov)
        t4 = time.perf_counter()
        ms = {"rectify": t1 - t0, "terrain": t2 - t1, "scene": t3 - t2, "map": t4 - t3}
        return FrameResult(t, left, terrain, scene, mapped, {k: 1000 * v for k, v in ms.items()})


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
