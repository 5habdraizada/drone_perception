"""Closed-loop mission simulation: perception -> map -> planner -> simple drone kinematics.

The drone flies straight at the commanded waypoint (speed-limited, climb-limited),
burns battery per metre and per second, and lands when told to. Perception is
any object with .frame(x, y, agl, t) and .pose(x, y, agl), e.g. OraclePerception.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from perception.decision import MissionPlanner
from perception.mapping import GridMap


@dataclass
class DroneSim:
    x: float
    y: float
    agl: float = 0.0
    battery: float = 1.0
    speed: float = 3.0            # m/s horizontal
    climb_rate: float = 1.5       # m/s vertical
    battery_per_m: float = 0.001
    battery_per_s: float = 0.0005  # hovering costs too
    pose_sigma: float = 0.1       # m; tests can make it grow

    def step(self, tx: float, ty: float, tagl: float, dt: float) -> float:
        """Move towards the target for dt seconds; returns metres flown horizontally."""
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        move = min(d, self.speed * dt)
        if d > 1e-9:
            self.x += dx / d * move
            self.y += dy / d * move
        dz = tagl - self.agl
        self.agl += max(-self.climb_rate * dt, min(self.climb_rate * dt, dz))
        self.battery -= move * self.battery_per_m + dt * self.battery_per_s
        return move


@dataclass
class MissionResult:
    trace: list[dict] = field(default_factory=list)
    landed: bool = False
    grid: GridMap | None = None
    planner: MissionPlanner | None = None


def run_mission(perception, grid: GridMap, planner: MissionPlanner, drone: DroneSim,
                max_steps: int = 2000, dt: float = 1.0, sigma_growth: float = 0.0) -> MissionResult:
    """sigma_growth: metres of position uncertainty added per second (e.g. drifting without fixes).
    If the planner has a belief map, frames feed it too."""
    result = MissionResult(grid=grid, planner=planner)
    for i in range(max_steps):
        t = i * dt
        if drone.agl > 0.5:
            frame, pose = perception.frame(drone.x, drone.y, drone.agl, t), perception.pose(drone.x, drone.y, drone.agl)
            grid.update(frame, pose)
            if planner.belief is not None:
                planner.belief.observe(frame, pose)
        cmd = planner.decide(grid, drone.x, drone.y, drone.agl, drone.battery, drone.pose_sigma, t)
        result.trace.append({"t": t, "x": drone.x, "y": drone.y, "agl": drone.agl, "battery": drone.battery,
                             "mode": cmd.mode, "reason": cmd.reason, "tx": cmd.x, "ty": cmd.y, "tagl": cmd.agl})
        if cmd.mode == "land":
            drone.step(cmd.x, cmd.y, 0.0, dt)
            if drone.agl <= 0.05:
                result.landed = True
                break
            continue
        drone.step(cmd.x, cmd.y, cmd.agl, dt)
        drone.pose_sigma += sigma_growth * dt
    return result
