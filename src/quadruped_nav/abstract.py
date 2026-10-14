"""Deterministic, uncalibrated navigation model; no rigid-body physics or learned policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import blake2s
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Pile:
    id: str
    x: float
    y: float
    z: float
    size: float
    roll: float = 0.0
    pitch: float = 0.0


@dataclass(frozen=True)
class Limits:
    observation_radius_m: float = 2.0
    max_subgoal_m: float = 2.0
    max_gap_m: float = 0.14
    max_height_step_m: float = 0.05
    speed_m_s: float = 0.6
    subgoal_budget_s: float = 4.0
    episode_budget_s: float = 60.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0 or (value == 0 and name not in {"max_gap_m", "max_height_step_m"}):
                raise ValueError(f"Invalid {name}: {value}")


@dataclass(frozen=True)
class Terrain:
    piles: tuple[Pile, ...]
    start_id: str
    goal_id: str
    start_heading: float = 0.0

    def __post_init__(self):
        ids = [p.id for p in self.piles]
        if len(ids) != len(set(ids)) or self.start_id not in ids or self.goal_id not in ids:
            raise ValueError("Terrain needs unique pile IDs and valid start/goal IDs")
        if not math.isfinite(self.start_heading):
            raise ValueError("Start heading must be finite")
        for p in self.piles:
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z, p.size,p.roll,p.pitch)) or p.size <= 0 or max(abs(p.roll),abs(p.pitch))>math.pi/4:
                raise ValueError("Pile geometry must be finite with positive size")

    @classmethod
    def from_scene(cls, scene: dict, target: str = "P2") -> Terrain:
        """Adapt the existing v1, horizontal square-pile schema without changing it."""
        if scene.get("schema_version") != 1 or scene.get("units") != "metres":
            raise ValueError("Expected a v1 scene in metres")
        def opaque(source_id):
            return "pile_" + blake2s(source_id.encode(), digest_size=8).hexdigest()
        records = scene["stones"] + scene["platforms"]
        for record in records:
            if record.get('yaw',0)!=0:
                raise ValueError("Yaw-rotated pile footprints are not supported")
        piles = tuple(Pile(opaque(p["id"]), float(p["x"]), float(p["y"]),
                           float(p["z_top"]), float(p["size"]),float(p.get('roll',0)),float(p.get('pitch',0))) for p in records)
        return cls(piles, opaque(scene["start"]["platform"]), opaque(target),
                   float(scene["start"].get("heading", 0)))


@dataclass(frozen=True)
class VisiblePile:
    id: str
    relative_top_center_m: tuple[float, float, float]
    top_size_xy_m: tuple[float, float]
    top_normal_local: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Observation:
    decision_index: int
    time_s: float
    current_pile_id: str
    position_odom_m: tuple[float, float, float]
    yaw_rad: float
    goal_relative_xy_m: tuple[float, float]
    observation_radius_m: float
    visible_piles: tuple[VisiblePile, ...]
    action_mask: tuple[bool, ...]


def straight_approach(start: Pile, target: Pile, supports: Sequence[Pile], limits: Limits) -> bool:
    """Check support intervals along the straight centreline, including every gap.

    This point-body model has no foot placement or body-clearance guarantee. It
    cannot turn along a winding graph path while pretending to follow one target.
    """
    length = math.hypot(target.x - start.x, target.y - start.y)
    if length < 1e-9:
        return False
    centres = np.array([(p.x, p.y) for p in supports])
    # Conservative inscribed square for tilted top footprints; no claim of
    # exact contact geometry in the abstract surrogate.
    halves = np.array([p.size / 2 * min(math.cos(p.roll),math.cos(p.pitch)) for p in supports])[:, None]
    delta = np.array([target.x - start.x, target.y - start.y])
    origin = np.array([start.x, start.y])
    low, high = centres - halves, centres + halves
    enter, leave = np.zeros(len(supports)), np.ones(len(supports))
    for axis in range(2):
        if abs(delta[axis]) < 1e-12:
            outside = (origin[axis] < low[:, axis]) | (origin[axis] > high[:, axis])
            leave[outside] = -1
        else:
            a = (low[:, axis] - origin[axis]) / delta[axis]
            b = (high[:, axis] - origin[axis]) / delta[axis]
            enter = np.maximum(enter, np.minimum(a, b))
            leave = np.minimum(leave, np.maximum(a, b))
    intervals = sorted(((float(enter[i] * length), float(leave[i] * length), p)
                        for i, p in enumerate(supports) if leave[i] - enter[i] > 1e-9),
                       key=lambda item: (item[1], item[0], item[2].id))
    reachable = [(hi, p) for _, hi, p in intervals if p.id == start.id]
    for lo, hi, pile in intervals:
        if pile.id == start.id or any(
            lo - prev_hi <= limits.max_gap_m + 1e-8
            and hi > prev_hi + 1e-9
            and abs(pile.z - prev.z) <= limits.max_height_step_m + 1e-8
            for prev_hi, prev in reachable
        ):
            reachable.append((hi, pile))
            if pile.id == target.id:
                return True
    return False


class Geometry:
    """Full-map geometry belongs to the environment and privileged oracle only."""
    def __init__(self, terrain: Terrain, limits: Limits):
        self.terrain, self.limits = terrain, limits
        self.by_id = {p.id: p for p in terrain.piles}
        self._cache: dict[str, tuple[tuple[Pile, ...], tuple[bool, ...]]] = {}

    def local(self, current_id: str):
        if current_id not in self._cache:
            current = self.by_id[current_id]
            distances = {p.id: math.hypot(p.x-current.x, p.y-current.y) for p in self.terrain.piles}
            visible = tuple(sorted((p for p in self.terrain.piles
                                    if distances[p.id] <= self.limits.observation_radius_m + 1e-9),
                                   key=lambda p: p.id))
            max_distance = min(self.limits.max_subgoal_m,
                               self.limits.speed_m_s * self.limits.subgoal_budget_s)
            mask = tuple(p.id != current_id and distances[p.id] <= max_distance + 1e-9
                         and straight_approach(current, p, visible, self.limits) for p in visible)
            self._cache[current_id] = (visible, mask)
        return self._cache[current_id]

    def neighbours(self, current_id: str):
        visible, mask = self.local(current_id)
        return tuple(p for p, allowed in zip(visible, mask) if allowed)


class AbstractEnv:
    """Each step executes one subgoal under a deterministic geometric surrogate."""
    def __init__(self, terrain: Terrain, limits: Limits | None = None):
        self.terrain = terrain
        self.limits = limits or Limits()
        self.geometry = Geometry(terrain, self.limits)
        self.reset()

    def reset(self):
        self.current_id = self.terrain.start_id
        self.yaw = self.terrain.start_heading
        self.time_s = 0.0
        self.distance_m = 0.0
        self.decisions = 0
        self.status = "success" if self.current_id == self.terrain.goal_id else "running"
        self.events = []
        self.states = [self.state()]
        return self.observe()

    def state(self):
        p = self.geometry.by_id[self.current_id]
        return {"pile_id": p.id, "x": p.x, "y": p.y, "z": p.z,
                "yaw_rad": self.yaw, "time_s": self.time_s, "distance_m": self.distance_m,
                "status": self.status}

    def observe(self):
        current = self.geometry.by_id[self.current_id]
        goal = self.geometry.by_id[self.terrain.goal_id]
        visible, mask = self.geometry.local(self.current_id)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        def local(x, y):
            dx, dy = x-current.x, y-current.y
            return c*dx+s*dy, -s*dx+c*dy
        return Observation(self.decisions, self.time_s, current.id,
                           (current.x, current.y, current.z + .3), self.yaw,
                           local(goal.x, goal.y), self.limits.observation_radius_m,
                           tuple(VisiblePile(p.id, (*local(p.x, p.y), p.z-current.z-.3),
                                             (p.size, p.size),local_normal(p,self.yaw)) for p in visible), mask)

    def step(self, target_id: str | None):
        if self.status != "running":
            raise RuntimeError("Reset before stepping a finished episode")
        observation = self.observe()
        valid_ids = {p.id for p, allowed in zip(observation.visible_piles, observation.action_mask) if allowed}
        if target_id is not None and target_id not in valid_ids:
            raise ValueError("Selected pile is not eligible in the current local observation")
        event = {"decision_index": self.decisions, "observation": asdict(observation),
                 "target_pile_id": target_id, "started_at_s": self.time_s}
        if target_id is None:
            self.status = "no_candidate" if not valid_ids else "planner_no_route"
        else:
            start, target = self.geometry.by_id[self.current_id], self.geometry.by_id[target_id]
            distance = math.hypot(target.x-start.x, target.y-start.y)
            duration = distance / self.limits.speed_m_s
            if self.time_s + duration > self.limits.episode_budget_s + 1e-9:
                # Atomic subgoal model: an unfinished transition leaves the robot
                # at its last confirmed support. No partial-motion physics claimed.
                self.time_s = self.limits.episode_budget_s
                self.status = "timeout"
            else:
                self.time_s += duration
                self.distance_m += distance
                self.yaw = math.atan2(target.y-start.y, target.x-start.x)
                self.current_id = target_id
                if target_id == self.terrain.goal_id:
                    self.status = "success"
                elif self.time_s >= self.limits.episode_budget_s - 1e-9:
                    self.status = "timeout"
        self.decisions += 1
        event.update(outcome=self.status if self.status != "running" else "arrived",
                     ended_at_s=self.time_s)
        self.events.append(event)
        self.states.append(self.state())
        return self.observe(), self.status != "running"


def local_normal(pile,yaw):
    from .terrain import surface_normal
    n=surface_normal(pile.roll,pile.pitch);c,s=math.cos(yaw),math.sin(yaw)
    return float(c*n[0]+s*n[1]),float(-s*n[0]+c*n[1]),float(n[2])


def observe_pose(terrain,limits,position,yaw,decision_index=0,time_s=0):
    """Planner observation at the actual continuous physics pose (not snapped)."""
    x,y,z=map(float,position);c,s=math.cos(yaw),math.sin(yaw)
    support=min(terrain.piles,key=lambda p:math.hypot(p.x-x,p.y-y))
    current=Pile(support.id,x,y,support.z,support.size,support.roll,support.pitch)
    visible=tuple(sorted((p for p in terrain.piles if math.hypot(p.x-x,p.y-y)<=limits.observation_radius_m+1e-9),key=lambda p:p.id))
    def local(px,py):return c*(px-x)+s*(py-y),-s*(px-x)+c*(py-y)
    candidates=tuple(math.hypot(p.x-x,p.y-y)>.25 and math.hypot(p.x-x,p.y-y)<=limits.max_subgoal_m and
                     straight_approach(current,p,visible,limits) for p in visible)
    goal=next(p for p in terrain.piles if p.id==terrain.goal_id)
    return Observation(decision_index,time_s,support.id,(x,y,z),yaw,local(goal.x,goal.y),limits.observation_radius_m,
                       tuple(VisiblePile(p.id,(*local(p.x,p.y),p.z-z),(p.size,p.size),local_normal(p,yaw)) for p in visible),candidates)
